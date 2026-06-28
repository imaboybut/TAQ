from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn



class FakeQuantizerBase(nn.Module):
    def __init__(self, bit: int = 4) -> None:
        super().__init__()
        self.bit = bit
        self.lb = torch.tensor(0.0)
        self.ub = torch.tensor(0.0)
        self.identity = False


    def set_n_bit_manually(self, bit: int) -> None:
        self.bit = bit

    def set_params_lb_manually(self, lb: float) -> None:
        device = self.lb.device
        self.lb = torch.tensor(float(lb), device=device)

    def set_params_ub_manually(self, ub: float) -> None:
        device = self.ub.device
        self.ub = torch.tensor(float(ub), device=device)

    def quantise(self, tensor: Tensor) -> Tensor:

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
        # Straight-through estimator: keep gradient wrt lb/ub/scale.
        normalised = normalised + (quantised - normalised).detach()
        return normalised * scale + lb

    def forward(self, tensor: Tensor) -> Tensor:
        return self.quantise(tensor)


class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self, bit: int = 4) -> None:
        super().__init__(bit=bit)


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self, bit: int = 4) -> None:
        super().__init__(bit=bit)


class QuantConv2d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        bit = config.get("bit", 4)
        self.weight = nn.Parameter(torch.empty(0))
        self.bias: Optional[nn.Parameter] = None
        self.kwargs: dict = {}
        self.quant = True
        self.weight_quantizer = FakeQuantizerWeight(bit=bit)
        self.act_quantizer = FakeQuantizerAct(bit=bit)

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
        out = F.conv2d(self.act_quantizer(x), w, self.bias, **self.kwargs)
        return out


class QuantLinear(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        bit = config.get("bit", 4)
        self.weight = nn.Parameter(torch.empty(0))
        self.bias: Optional[nn.Parameter] = None
        self.quant = True
        self.weight_quantizer = FakeQuantizerWeight(bit=bit)
        self.act_quantizer = FakeQuantizerAct(bit=bit)

    def set_param(self, linear: nn.Linear) -> None:
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None


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
