#!/usr/bin/env python3
"""Inference-only entrypoint that consumes a saved quantised RealViformer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import cv2
import torch

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archs.realviformer_arch import RealViformer  # type: ignore  # noqa: E402
from data_util import read_img_seq  # type: ignore  # noqa: E402
from img_util import tensor2img  # type: ignore  # noqa: E402

from TAQ import TAQ  # noqa: E402

def load_state_dict(fp_model: RealViformer, model_path: str) -> None:
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        if "params" in state:
            state = state["params"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unrecognised checkpoint format at {model_path}")
    state = dict(state)
    if state.pop("attn_merge.attn.masktemp", None) is not None:
        print("[INFO] Ignored legacy buffer: attn_merge.attn.masktemp")
    missing, unexpected = fp_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


def parse_sequence_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def collect_frame_paths(directory: Path) -> List[str]:
    if not directory.exists():
        raise FileNotFoundError(f"Sequence directory {directory} does not exist.")
    if directory.is_file():
        raise ValueError(f"{directory} must be a directory containing frames.")
    frames = sorted(
        str(p)
        for p in directory.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frames:
        raise FileNotFoundError(f"No frames found in {directory}")
    return frames


def _iter_sequence_dirs(root: Path):
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        yield entry


def gather_sequences(root: Path, names: Sequence[str]) -> Dict[str, List[str]]:
    if names:
        sequences: Dict[str, List[str]] = {}
        for name in names:
            seq_dir = root / name
            if not seq_dir.exists():
                raise FileNotFoundError(f"Sequence {name} missing under {root}")
            if seq_dir.is_file():
                raise ValueError(f"{seq_dir} must be a directory.")
            sequences[name] = collect_frame_paths(seq_dir)
        return sequences
    subdirs = list(_iter_sequence_dirs(root))
    if not subdirs:
        if root.name.startswith("."):
            raise FileNotFoundError(f"No valid sequences present under {root}")
        return {root.name: collect_frame_paths(root)}
    sequences: Dict[str, List[str]] = {}
    for subdir in subdirs:
        frames = collect_frame_paths(subdir)
        if frames:
            sequences[subdir.name] = frames
    if not sequences:
        raise FileNotFoundError(f"No sequences present under {root}")
    return sequences


def resolve_path(path_str: str, *, must_exist: bool) -> Path:
    """Resolve paths relative to the current working directory."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve(strict=False)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path '{path}' does not exist.")
    return path


def forward_sequence_with_padding(frames: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    with torch.inference_mode():
        padded = False
        _, _, _, h, w = frames.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            padded = True
            frames = torch.nn.functional.pad(
                frames.squeeze(0),
                pad=(pad_w, 0, pad_h, 0),
                mode="reflect",
            ).unsqueeze(0)
        outputs = model(frames)
    if padded:
        outputs = outputs[..., pad_h * 4 :, pad_w * 4 :]
    return outputs.squeeze(0)


def _coerce_bounds(bounds):
    if bounds is None:
        return None
    if isinstance(bounds, dict):
        lb = bounds.get("lb")
        ub = bounds.get("ub")
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lb, ub = bounds
    else:
        return None
    if lb is None or ub is None:
        return None
    return (float(lb), float(ub))


def _parse_range_blob(blob) -> Dict[str, Tuple[float, float]]:
    parsed: Dict[str, Tuple[float, float]] = {}
    if not isinstance(blob, dict):
        return parsed
    for name, bounds in blob.items():
        pair = _coerce_bounds(bounds)
        if pair is None:
            continue
        parsed[name] = pair
    return parsed


def _apply_ranges(wrapper: TAQ, ranges: Dict[str, Tuple[float, float]], *, kind: str) -> None:
    for name, module in wrapper.quant_modules.items():
        getter_name = "get_act_quantizer" if kind == "act" else "get_weight_quantizer"
        getter = getattr(module, getter_name, None)
        if getter is None:
            continue
        quantizer = getter()
        if quantizer is None:
            continue
        bounds = ranges.get(name)
        if bounds is None:
            continue
        lb, ub = bounds
        if hasattr(quantizer, "set_params_lb_manually"):
            quantizer.set_params_lb_manually(lb)
        else:
            quantizer.lb = torch.tensor(lb, device=quantizer.lb.device)
        if hasattr(quantizer, "set_params_ub_manually"):
            quantizer.set_params_ub_manually(ub)
        else:
            quantizer.ub = torch.tensor(ub, device=quantizer.ub.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional FP32 checkpoint; omit to rely solely on --quant_state_path.",
    )
    parser.add_argument("--quant_state_path", type=str, required=True, help="Checkpoint exported by save_pth.py")
    parser.add_argument("--lq_root", type=str, required=True, help="Directory with input LQ frame folders.")
    parser.add_argument("--sequences", type=str, default=None, help="Comma-separated subset of sequence names.")
    parser.add_argument(
        "--save_root",
        type=str,
        default="results/ours_inference",
        help="Where to save the inferred frames (one subfolder per sequence).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Frames per chunk (<=0 disables chunking and processes whole sequences).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--no_quant_conv", action="store_true")
    parser.add_argument("--no_quant_linear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    lq_root = resolve_path(args.lq_root, must_exist=True)
    sequence_names = parse_sequence_list(args.sequences)
    lq_sequences = gather_sequences(lq_root, sequence_names)

    save_root = resolve_path(args.save_root, must_exist=False)
    save_root.mkdir(parents=True, exist_ok=True)

    fp_model = RealViformer(
        num_feat=48,
        num_blocks=[2, 3, 4, 1],
        spynet_path=None,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        merge_head=2,
        bias=False,
        LayerNorm_type="BiasFree",
        ch_compress=True,
        squeeze_factor=[4, 4, 4],
        masked=True,
    )
    if args.model_path:
        model_path = resolve_path(args.model_path, must_exist=True)
        load_state_dict(fp_model, str(model_path))
    else:
        print("[INFO] --model_path not provided, skipping FP32 checkpoint load.")

    quant_wrapper = TAQ(
        fp_model,
        device=device,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        quantize_conv=not args.no_quant_conv,
        quantize_linear=not args.no_quant_linear,
    )
    print(f"[INFO] Injected {len(quant_wrapper.quant_modules)} quant module(s).")

    quant_state_path = resolve_path(args.quant_state_path, must_exist=True)
    checkpoint = torch.load(str(quant_state_path), map_location="cpu")
    act_ranges_blob = None
    weight_ranges_blob = None
    state = checkpoint
    if isinstance(checkpoint, dict):
        act_ranges_blob = checkpoint.get("act_ranges") or checkpoint.get("ranges")
        weight_ranges_blob = checkpoint.get("weight_ranges")
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "params" in checkpoint:
            state = checkpoint["params"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {args.quant_state_path}")
    state = dict(state)
    state.pop("attn_merge.attn.masktemp", None)
    missing, unexpected = quant_wrapper.quant_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing {len(missing)} parameter(s) while loading quant checkpoint; first: {missing[:5]}")
    if unexpected:
        print(
            f"[WARN] Unexpected {len(unexpected)} parameter(s) while loading quant checkpoint; first: {unexpected[:5]}"
        )

    act_ranges = _parse_range_blob(act_ranges_blob)
    weight_ranges = _parse_range_blob(weight_ranges_blob)
    if act_ranges:
        print(f"[INFO] Restored {len(act_ranges)} activation range(s) from checkpoint.")
        _apply_ranges(quant_wrapper, act_ranges, kind="act")
    else:
        print("[WARN] No activation ranges bundled in quant checkpoint; activations may be over-clipped.")
    if weight_ranges:
        print(f"[INFO] Restored {len(weight_ranges)} weight range(s) from checkpoint.")
        _apply_ranges(quant_wrapper, weight_ranges, kind="weight")
    quant_model = quant_wrapper.quant_model.to(device).eval()

    print(f"[INFER] Running {len(lq_sequences)} sequence(s). Output -> {save_root}")
    for seq_name, frame_paths in lq_sequences.items():
        seq_dir = save_root / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFER] Sequence {seq_name}: {len(frame_paths)} frame(s)")

        preds_written = 0
        chunk_size = args.interval if args.interval > 0 else len(frame_paths)
        chunks: List[List[str]]
        if len(frame_paths) <= chunk_size:
            chunks = [frame_paths]
        else:
            chunks = [
                frame_paths[i : i + chunk_size]
                for i in range(0, len(frame_paths), chunk_size)
            ]

        for chunk_idx, chunk in enumerate(chunks, start=1):
            clip, _ = read_img_seq(chunk, return_imgname=True)
            clip = clip.unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs = forward_sequence_with_padding(clip, quant_model)
            outputs_cpu = outputs.detach().cpu()
            for frame in outputs_cpu:
                img = tensor2img(frame, rgb2bgr=False)
                save_path = seq_dir / f"{preds_written:08d}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                preds_written += 1
            print(f"  [INFER] chunk {chunk_idx}/{len(chunks)} -> {preds_written} frame(s)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
