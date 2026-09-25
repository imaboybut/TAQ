#!/usr/bin/env python3
"""End-to-end TAQ calibration (paper Algorithm 1) and static checkpoint export.

For every calibration sequence:
  1. Sequence-wise histogram initialisation of the activation bounds (Sec. 3.2).
     Weight bounds are MSE-initialised when the quantised layers are built.
  2. Temporal refinement of all (lb, ub) with the frame-difference loss (Sec. 3.3).
The per-sequence bounds are then ensembled by arithmetic mean (Sec. 3.4) and
exported as one static checkpoint that ``inference.py`` consumes.

Paper setting (RealViformer): 9 REDS sequences (val 021-029) x 20 frames,
K=1024 histogram bins, 0.1% / 99.9% percentile clipping, Adam lr 5e-4, 5 epochs.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_util import read_img_seq  # type: ignore
from inference import (
    forward_sequence_with_padding,
    gather_sequences,
    load_state_dict,
    parse_sequence_list,
    resolve_path,
)
from TAQ import TAQ, Ranges, build_realviformer


# ---------------------------------------------------------------------------
# Temporal refinement of quantiser bounds (Sec. 3.3)
# ---------------------------------------------------------------------------
def _forward_single_frame(
    frame: torch.Tensor,
    model: torch.nn.Module,
    *,
    device: torch.device,
    requires_grad: bool,
) -> torch.Tensor:
    """Run ``model`` on one frame ``(1, C, H, W)``; returns the SR frame ``(C, 4H, 4W)``."""
    frames = frame.unsqueeze(0)  # (1, 1, C, H, W)
    grad_ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with grad_ctx:
        _, _, _, h, w = frames.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            frames = F.pad(frames.squeeze(0), pad=(pad_w, 0, pad_h, 0), mode="reflect").unsqueeze(0)
        outputs = model(frames.to(device))
    if pad_h or pad_w:
        outputs = outputs[..., pad_h * 4 :, pad_w * 4 :]
    return outputs[0, 0]


def refine_bounds_temporal(
    wrapper: TAQ,
    fp_model: torch.nn.Module,
    clip: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    """Minimise the temporal consistency loss of Eq. (4) w.r.t. every (lb, ub).

    The FP model and the quantised model are run frame by frame. From the second
    frame on, each Adam step aligns the quantised frame difference Q_t - Q_{t-1}
    with the FP frame difference F_t - F_{t-1}. Weights and bit-widths are fixed;
    only the clipping bounds of the weight and activation quantisers are learned.
    """
    if epochs <= 0 or clip.shape[0] < 2:
        return

    params: List[torch.nn.Parameter] = []
    quantizers: List[Tuple[str, torch.nn.Module]] = []
    for module in wrapper.quant_modules.values():
        for kind, quantizer in (("act", module.get_act_quantizer()), ("weight", module.get_weight_quantizer())):
            if not quantizer.calibrated:
                continue
            quantizer.lb = torch.nn.Parameter(torch.as_tensor(quantizer.lb).detach().clone().to(device))
            quantizer.ub = torch.nn.Parameter(torch.as_tensor(quantizer.ub).detach().clone().to(device))
            params.extend([quantizer.lb, quantizer.ub])
            quantizers.append((kind, quantizer))
    if not params:
        return

    def _span_stats() -> Dict[str, float]:
        stats: Dict[str, List[float]] = {"act": [], "weight": []}
        for kind, quantizer in quantizers:
            stats[kind].append(float((quantizer.ub - quantizer.lb).detach().cpu().item()))
        return {kind: float(np.mean(v)) for kind, v in stats.items() if v}

    span_before = _span_stats()
    optimizer = torch.optim.Adam(params, lr=lr)
    mse_loss = torch.nn.MSELoss()
    num_frames = clip.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0.0
        prev_fp: Optional[torch.Tensor] = None
        prev_q: Optional[torch.Tensor] = None
        for t in range(num_frames):
            frame = clip[t : t + 1]
            fp_frame = _forward_single_frame(frame, fp_model, device=device, requires_grad=False)

            optimizer.zero_grad()
            q_frame = _forward_single_frame(frame, wrapper.quant_model, device=device, requires_grad=True)
            if prev_fp is None or prev_q is None:
                prev_fp = fp_frame.detach()
                prev_q = q_frame.detach()
                continue

            loss = mse_loss(q_frame - prev_q, fp_frame - prev_fp)
            loss.backward()
            optimizer.step()

            # Keep lb < ub (minimum span 1e-6) after every update.
            with torch.no_grad():
                for _, quantizer in quantizers:
                    span = torch.clamp(quantizer.ub.data - quantizer.lb.data, min=1e-6)
                    center = 0.5 * (quantizer.ub.data + quantizer.lb.data)
                    quantizer.lb.data.copy_(center - 0.5 * span)
                    quantizer.ub.data.copy_(center + 0.5 * span)

            # The previous quantised frame is recomputed with the updated bounds.
            prev_q = _forward_single_frame(frame, wrapper.quant_model, device=device, requires_grad=False).detach()
            prev_fp = fp_frame.detach()
            epoch_loss += float(loss.detach().cpu().item())

        print(f"[REFINE] epoch {epoch + 1}/{epochs} avg_temporal_loss={epoch_loss / (num_frames - 1):.6e}")

    for _, quantizer in quantizers:
        quantizer.lb.requires_grad_(False)
        quantizer.ub.requires_grad_(False)
    span_after = _span_stats()
    for kind in span_after:
        print(f"[REFINE] mean {kind} span {span_before[kind]:.6f} -> {span_after[kind]:.6f}")


@torch.no_grad()
def compute_temporal_losses(
    fp_model: torch.nn.Module,
    quant_model: torch.nn.Module,
    clip: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[float, float]:
    """Pixel MSE and frame-difference MSE between FP and quantised outputs (diagnostic)."""
    fp_out = forward_sequence_with_padding(clip.to(device), fp_model)
    q_out = forward_sequence_with_padding(clip.to(device), quant_model)
    pixel = torch.mean((fp_out - q_out) ** 2)
    if fp_out.shape[0] > 1:
        temporal = torch.mean(((fp_out[1:] - fp_out[:-1]) - (q_out[1:] - q_out[:-1])) ** 2)
    else:
        temporal = torch.zeros(())
    return float(pixel.cpu().item()), float(temporal.cpu().item())


# ---------------------------------------------------------------------------
# Sequence-wise bounds ensembling (Sec. 3.4)
# ---------------------------------------------------------------------------
def mean_ranges(entries: List[dict], key: str) -> Ranges:
    """Arithmetic mean of the per-sequence (lb, ub) for each quantiser (Eq. (5))."""
    collected: Dict[str, List[Tuple[float, float]]] = {}
    for entry in entries:
        for name, bounds in entry[key].items():
            collected.setdefault(name, []).append((float(bounds["lb"]), float(bounds["ub"])))
    return {
        name: (float(np.mean([lb for lb, _ in pairs])), float(np.mean([ub for _, ub in pairs])))
        for name, pairs in collected.items()
    }


def _serialise(ranges: Ranges) -> Dict[str, Dict[str, float]]:
    return {name: {"lb": lb, "ub": ub} for name, (lb, ub) in ranges.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str, required=True, help="FP32 RealViformer checkpoint.")
    parser.add_argument("--calib_root", type=str, required=True, help="Directory with one LQ frame folder per calibration sequence.")
    parser.add_argument("--calib_sequences", type=str, default=None, help="Comma-separated sequence names (default: every sub-folder).")
    parser.add_argument("--calib_sequence_limit", type=int, default=0, help="Use only the first N sequences (<=0: all).")
    parser.add_argument("--calib_frames", type=int, default=20, help="Frames per sequence for initialisation and refinement (<=0: all).")
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--range_ft_epochs", type=int, default=5, help="Temporal refinement epochs per sequence (0 disables).")
    parser.add_argument("--range_ft_lr", type=float, default=5e-4, help="Adam learning rate for the refinement.")
    parser.add_argument("--calib_clip_low", type=float, default=0.001, help="Lower percentile for activation clipping before the histogram.")
    parser.add_argument("--calib_clip_high", type=float, default=0.999, help="Upper percentile for activation clipping before the histogram.")
    parser.add_argument("--disable_calib_clip", action="store_true", help="No percentile clipping (the 'no clip' ablation).")
    parser.add_argument("--no_quant_conv", action="store_true")
    parser.add_argument("--no_quant_linear", action="store_true")
    parser.add_argument(
        "--skip_io_layers",
        action="store_true",
        help="Keep shallow_extraction.0 and conv_last in FP32 (default: all layers quantised, as in the paper).",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Where per-sequence and averaged ranges (JSON) are written.")
    parser.add_argument("--save_path", type=str, default=None, help="Exported checkpoint path (default: <out_dir>/TAQ_w{w}a{a}.pth).")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.disable_calib_clip:
        clip_low: Optional[float] = None
        clip_high: Optional[float] = None
    else:
        clip_low, clip_high = float(args.calib_clip_low), float(args.calib_clip_high)

    calib_root = resolve_path(args.calib_root, must_exist=True)
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = gather_sequences(calib_root, parse_sequence_list(args.calib_sequences))
    if args.calib_sequence_limit > 0:
        sequences = dict(list(sequences.items())[: args.calib_sequence_limit])

    quant_kwargs = dict(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        quantize_conv=not args.no_quant_conv,
        quantize_linear=not args.no_quant_linear,
        skip_io_layers=args.skip_io_layers,
        device=device,
    )
    print(
        f"[CONFIG] sequences={list(sequences)} frames={args.calib_frames} "
        f"bits=W{args.w_bit}/A{args.a_bit} refine_epochs={args.range_ft_epochs} refine_lr={args.range_ft_lr} "
        f"clip={'off' if clip_low is None else f'{clip_low}-{clip_high}'} skip_io_layers={args.skip_io_layers}"
    )

    fp_model = build_realviformer()
    load_state_dict(fp_model, str(resolve_path(args.model_path, must_exist=True)))
    fp_model = fp_model.to(device).eval()

    # Stage 1 + 2: per-sequence initialisation and temporal refinement.
    entries: List[dict] = []
    for seq_name, frame_paths in sequences.items():
        frames = frame_paths[: args.calib_frames] if args.calib_frames > 0 else frame_paths
        clip, _ = read_img_seq(frames, return_imgname=True)  # (T, C, H, W), RGB in [0, 1]
        print(f"[CALIB] {seq_name}: {len(frames)} frame(s) of size {tuple(clip.shape[-2:])}")

        wrapper = TAQ(deepcopy(fp_model), calib_clip_low=clip_low, calib_clip_high=clip_high, **quant_kwargs)
        if not entries:
            print(f"[CALIB] Injected {len(wrapper.quant_modules)} quantised layer(s).")
        wrapper.calibrate([clip])
        refine_bounds_temporal(wrapper, fp_model, clip, device=device, epochs=args.range_ft_epochs, lr=args.range_ft_lr)

        pixel_loss, temporal_loss = compute_temporal_losses(fp_model, wrapper.quant_model.eval(), clip, device=device)
        print(f"[CALIB] {seq_name}: pixel_mse={pixel_loss:.6f} temporal_mse={temporal_loss:.6f}")

        entry = {
            "sequence": seq_name,
            "frames_used": len(frames),
            "ranges": wrapper.extract_ranges("act"),
            "weight_ranges": wrapper.extract_ranges("weight"),
            "pixel_loss": pixel_loss,
            "temporal_loss": temporal_loss,
        }
        entry_path = out_dir / f"{seq_name}_frames0-{len(frames)}_ranges.json"
        with open(entry_path, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2)
        entries.append(entry)

        del wrapper
        torch.cuda.empty_cache()

    if not entries:
        raise RuntimeError("No calibration sequence was processed.")

    # Stage 3: mean ensembling over sequences.
    act_ranges = mean_ranges(entries, "ranges")
    weight_ranges = mean_ranges(entries, "weight_ranges")
    averaged_path = out_dir / "averaged_ranges.json"
    with open(averaged_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "calibration": {
                    "w_bit": args.w_bit,
                    "a_bit": args.a_bit,
                    "frames_per_sequence": args.calib_frames,
                    "clip_low": clip_low,
                    "clip_high": clip_high,
                    "range_ft_epochs": args.range_ft_epochs,
                    "range_ft_lr": args.range_ft_lr,
                    "skip_io_layers": args.skip_io_layers,
                },
                "aggregation": {"method": "mean", "num_sequences": len(entries)},
                "sequences": [entry["sequence"] for entry in entries],
                "ranges": _serialise(act_ranges),
                "weight_ranges": _serialise(weight_ranges),
                "entries": entries,
            },
            fh,
            indent=2,
        )
    print(f"[ENSEMBLE] mean of {len(entries)} sequence(s) written to {averaged_path}")

    # Export one static checkpoint for inference.py.
    export = TAQ(deepcopy(fp_model), **quant_kwargs)
    missing = export.apply_ranges(act_ranges, "act") + export.apply_ranges(weight_ranges, "weight")
    if missing:
        print(f"[WARN] {len(missing)} quantiser(s) without ensembled ranges stay identity; first: {missing[:5]}")
    save_path = resolve_path(
        args.save_path or str(out_dir / f"TAQ_w{args.w_bit}a{args.a_bit}.pth"), must_exist=False
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in export.quant_model.state_dict().items()},
            "act_ranges": _serialise(act_ranges),
            "weight_ranges": _serialise(weight_ranges),
            "meta": {
                "w_bit": args.w_bit,
                "a_bit": args.a_bit,
                "quantize_conv": not args.no_quant_conv,
                "quantize_linear": not args.no_quant_linear,
                "skip_io_layers": args.skip_io_layers,
                "calib_dir": str(out_dir),
                "model_path": args.model_path,
            },
        },
        save_path,
    )
    print(f"[EXPORT] Static quantised checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
