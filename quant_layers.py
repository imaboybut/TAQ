"""Fake-quantised Conv2d / Linear wrappers using the affine quantiser of paper Sec. 3.1."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from calibration import search_weight_bounds


class FakeQuantizerBase(nn.Module):
    """Uniform asymmetric (affine) fake quantiser Q_b(v; lb, ub) of Eq. (1).

    ``calibrated`` is False until bounds have been installed. An uncalibrated
    quantiser, or one with ``identity`` set, passes its input through unchanged
    (this is how activations are captured during calibration).
    """

    def __init__(self, bit: int = 4) -> None:
        super().__init__()
        self.bit = bit
        self.lb = torch.tensor(0.0)
        self.ub = torch.tensor(0.0)
        self.identity = False
        self.calibrated = False

    def set_params_lb_manually(self, lb: float) -> None:
        device = self.lb.device
        self.lb = torch.tensor(float(lb), device=device)

    def set_params_ub_manually(self, ub: float) -> None:
        device = self.ub.device
        self.ub = torch.tensor(float(ub), device=device)

    def quantise(self, tensor: Tensor) -> Tensor:
        if self.identity or not self.calibrated:
            return tensor
        lb = self.lb.to(tensor.device, tensor.dtype)
        ub = self.ub.to(tensor.device, tensor.dtype)
        levels = (1 << self.bit) - 1
        span = ub - lb
        span = torch.where(span == 0, torch.ones_like(span), span)
        scale = span / max(levels, 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalised = (tensor - lb) / scale
        clipped = torch.clamp(normalised, 0.0, float(levels))
        quantised = torch.round(clipped)
        # Identity straight-through estimator: gradients reach lb/ub/scale.
        normalised = normalised + (quantised - normalised).detach()
        return normalised * scale + lb

    def forward(self, tensor: Tensor) -> Tensor:
        return self.quantise(tensor)


class FakeQuantizerWeight(FakeQuantizerBase):
    def calibrate(self, tensor: Tensor) -> None:
        """MSE-optimal (lb, ub) for a weight tensor (Algorithm 1, line 3)."""
        arr = tensor.detach().cpu().numpy()
        lb, ub = search_weight_bounds(arr, self.bit)
        self.lb = torch.tensor(lb, device=tensor.device, dtype=tensor.dtype)
        self.ub = torch.tensor(ub, device=tensor.device, dtype=tensor.dtype)
        self.calibrated = True


class FakeQuantizerAct(FakeQuantizerBase):
    pass


class QuantConv2d(nn.Module):
    def __init__(self, w_bit: int, a_bit: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0))
        self.bias: Optional[nn.Parameter] = None
        self.kwargs: dict = {}
        self.quant = True
        self.weight_quantizer = FakeQuantizerWeight(bit=w_bit)
        self.act_quantizer = FakeQuantizerAct(bit=a_bit)

    def set_param(self, conv: nn.Conv2d) -> None:
        self.weight = nn.Parameter(conv.weight.detach().clone())
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().clone())
        else:
            self.bias = None
        self.kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        }
        self.weight_quantizer.calibrate(self.weight)

    def set_quant_flag(self, enable: bool) -> None:
        self.quant = enable

    def get_weight_quantizer(self) -> FakeQuantizerWeight:
        return self.weight_quantizer

    def get_act_quantizer(self) -> FakeQuantizerAct:
        return self.act_quantizer

    def forward(self, x: Tensor) -> Tensor:
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.kwargs)
        w = self.weight_quantizer.quantise(self.weight)
        return F.conv2d(self.act_quantizer(x), w, self.bias, **self.kwargs)


class QuantLinear(nn.Module):
    def __init__(self, w_bit: int, a_bit: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0))
        self.bias: Optional[nn.Parameter] = None
        self.quant = True
        self.weight_quantizer = FakeQuantizerWeight(bit=w_bit)
        self.act_quantizer = FakeQuantizerAct(bit=a_bit)

    def set_param(self, linear: nn.Linear) -> None:
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None
        self.weight_quantizer.calibrate(self.weight)

    def set_quant_flag(self, enable: bool) -> None:
        self.quant = enable

    def get_weight_quantizer(self) -> FakeQuantizerWeight:
        return self.weight_quantizer

    def get_act_quantizer(self) -> FakeQuantizerAct:
        return self.act_quantizer

    def forward(self, x: Tensor) -> Tensor:
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        w = self.weight_quantizer.quantise(self.weight)
        return F.linear(self.act_quantizer(x), w, self.bias)
