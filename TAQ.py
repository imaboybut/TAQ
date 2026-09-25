"""TAQ model wrapper.

Replaces every Conv2d / Linear layer of a floating-point RealViformer with a
fake-quantised counterpart (weights are MSE-calibrated on construction) and
implements the sequence-wise activation calibration of paper Sec. 3.2:
activation statistics are captured with forward hooks while the activation
quantisers run as identity, then the histogram-MSE bounds are installed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from archs.realviformer_arch import RealViformer  # type: ignore
from calibration import MSEQuantCalibrator
from quant_layers import FakeQuantizerBase, QuantConv2d, QuantLinear

# Official RealViformer configuration used for every experiment in the paper.
REALVIFORMER_KWARGS = dict(
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

# Layers kept in floating point when ``skip_io_layers=True``. By default (the
# paper setting) every Conv/Linear layer is quantised, these two included.
IO_LAYER_NAMES = ("shallow_extraction.0", "conv_last")

QuantModule = Union[QuantConv2d, QuantLinear]
Ranges = Dict[str, Tuple[float, float]]


def build_realviformer() -> RealViformer:
    return RealViformer(**REALVIFORMER_KWARGS)


class TAQ(nn.Module):
    def __init__(
        self,
        fp_model: RealViformer,
        *,
        w_bit: int = 4,
        a_bit: int = 4,
        quantize_conv: bool = True,
        quantize_linear: bool = True,
        skip_io_layers: bool = False,
        calib_clip_low: Optional[float] = 0.001,
        calib_clip_high: Optional[float] = 0.999,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.fp_model = fp_model.to(self.device).eval()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.quantize_conv = quantize_conv
        self.quantize_linear = quantize_linear
        self.skip_names = set(IO_LAYER_NAMES) if skip_io_layers else set()
        self.calib_clip_low = calib_clip_low
        self.calib_clip_high = calib_clip_high

        self.quant_model = deepcopy(self.fp_model).to(self.device).eval()
        self.quant_modules: Dict[str, QuantModule] = {}
        self._inject_quant_modules()

        self._capture_hooks: List = []
        self._calibrators: Dict[int, MSEQuantCalibrator] = {}
        self._original_states: Dict[int, Tuple[bool, bool]] = {}

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.quant_model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Layer injection
    # ------------------------------------------------------------------
    def _inject_quant_modules(self) -> None:
        def convert(module: nn.Module, prefix: str = "") -> None:
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in self.skip_names:
                    continue
                if self.quantize_conv and isinstance(child, nn.Conv2d):
                    q: QuantModule = QuantConv2d(self.w_bit, self.a_bit)
                elif self.quantize_linear and isinstance(child, nn.Linear):
                    q = QuantLinear(self.w_bit, self.a_bit)
                else:
                    convert(child, full_name)
                    continue
                q.set_param(child)
                q = q.to(child.weight.device)
                setattr(module, name, q)
                self.quant_modules[full_name] = q

        convert(self.quant_model)
        self.quant_model.to(self.device)

    # ------------------------------------------------------------------
    # Range (lb, ub) import / export
    # ------------------------------------------------------------------
    @staticmethod
    def _quantizer(module: QuantModule, kind: str) -> FakeQuantizerBase:
        if kind == "act":
            return module.get_act_quantizer()
        if kind == "weight":
            return module.get_weight_quantizer()
        raise ValueError(f"kind must be 'act' or 'weight', got {kind!r}")

    def extract_ranges(self, kind: str) -> Dict[str, Dict[str, float]]:
        """Return ``{module_name: {"lb", "ub"}}`` for every calibrated quantiser of ``kind``."""
        ranges: Dict[str, Dict[str, float]] = {}
        for name, module in self.quant_modules.items():
            quantizer = self._quantizer(module, kind)
            if not quantizer.calibrated:
                continue
            ranges[name] = {
                "lb": float(quantizer.lb.detach().cpu().item()),
                "ub": float(quantizer.ub.detach().cpu().item()),
            }
        return ranges

    def apply_ranges(self, ranges: Ranges, kind: str) -> List[str]:
        """Install ``(lb, ub)`` per module name. Returns the names missing from ``ranges``."""
        missing: List[str] = []
        for name, module in self.quant_modules.items():
            bounds = ranges.get(name)
            if bounds is None:
                missing.append(name)
                continue
            quantizer = self._quantizer(module, kind)
            quantizer.set_params_lb_manually(bounds[0])
            quantizer.set_params_ub_manually(bounds[1])
            quantizer.calibrated = True
        return missing

    # ------------------------------------------------------------------
    # Sequence-wise histogram initialisation (Sec. 3.2)
    # ------------------------------------------------------------------
    def _begin_capture(self) -> None:
        if self._capture_hooks:
            return
        self._calibrators = {}
        self._original_states = {}

        for module in self.quant_modules.values():
            fq = module.get_act_quantizer()
            key = id(fq)
            self._calibrators[key] = MSEQuantCalibrator(
                bit=self.a_bit,
                clip_quantile_low=self.calib_clip_low,
                clip_quantile_high=self.calib_clip_high,
            )
            self._original_states[key] = (fq.identity, fq.calibrated)
            fq.identity = True
            fq.calibrated = True

            def _hook(_module, inputs, _output, key=key):
                if inputs and inputs[0] is not None:
                    self._calibrators[key].observe(inputs[0])

            self._capture_hooks.append((fq, fq.register_forward_hook(_hook)))

    def _finalise_capture(self) -> None:
        sample_counts: List[int] = []
        for fq, handle in self._capture_hooks:
            handle.remove()
            fq.identity, fq.calibrated = self._original_states[id(fq)]
            calibrator = self._calibrators[id(fq)]
            try:
                lb, ub = calibrator.finalise()
            except ValueError:  # quantiser never received data: stays identity
                continue
            sample_counts.append(int(calibrator.observer.total_count))
            fq.set_params_lb_manually(lb)
            fq.set_params_ub_manually(ub)
            fq.calibrated = True
        self._calibrators = {}
        self._original_states = {}
        self._capture_hooks.clear()

        if sample_counts:
            print(
                "[CALIB] Samples per activation quantizer -> "
                f"min={min(sample_counts)}, max={max(sample_counts)}, "
                f"avg={sum(sample_counts) / len(sample_counts):.1f}, total={sum(sample_counts)}"
            )

    @torch.no_grad()
    def calibrate(self, clips: Iterable[torch.Tensor]) -> None:
        """Initialise the activation bounds from ``clips`` (Sec. 3.2).

        Each clip is a ``(T, C, H, W)`` or ``(1, T, C, H, W)`` tensor in [0, 1].
        Every clip passed in one call feeds the same histograms, so call this
        once per calibration sequence to obtain sequence-specific bounds.
        """
        self.quant_model.eval()
        self._begin_capture()
        try:
            for clip in clips:
                if clip.dim() == 4:
                    clip = clip.unsqueeze(0)
                self.quant_model(clip.to(self.device))
        finally:
            torch.cuda.empty_cache()
            self._finalise_capture()
