from __future__ import annotations

from copy import deepcopy
from typing import Dict, Union

import torch
from torch import nn

from archs.realviformer_arch import RealViformer  # type: ignore  # noqa: E402
from quant_layers import QuantConv2d, QuantLinear


class TAQ(nn.Module):
    def __init__(
        self,
        fp_model: RealViformer,
        *,
        w_bit: int = 4,
        a_bit: int = 4,
        quantize_conv: bool = True,
        quantize_linear: bool = True,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.fp_model = fp_model.to(self.device).eval()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.quantize_conv = quantize_conv
        self.quantize_linear = quantize_linear

        self.quant_model = deepcopy(self.fp_model).to(self.device).eval()
        self.quant_modules: Dict[str, Union[QuantConv2d, QuantLinear]] = {}
        self._inject_quant_modules()

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.quant_model(*args, **kwargs)

    def _inject_quant_modules(self) -> None:
        config = {"bit": max(self.w_bit, self.a_bit)}

        def convert(module: nn.Module, prefix: str = "") -> None:
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                if self.quantize_conv and isinstance(child, nn.Conv2d):
                    q = QuantConv2d(config)
                    q.set_param(child)
                    q = q.to(child.weight.device)
                    q.set_quant_flag(True)
                    self._apply_bits(q)
                    setattr(module, name, q)
                    self.quant_modules[full_name] = q
                elif self.quantize_linear and isinstance(child, nn.Linear):
                    q = QuantLinear(config)
                    q.set_param(child)
                    q = q.to(child.weight.device)
                    q.set_quant_flag(True)
                    self._apply_bits(q)
                    setattr(module, name, q)
                    self.quant_modules[full_name] = q
                else:
                    convert(child, full_name)

        convert(self.quant_model)
        self.quant_model.to(self.device)
        
    def _apply_bits(self, module: Union[QuantConv2d, QuantLinear]) -> None:
        wq_getter = getattr(module, "get_weight_quantizer", None)
        aq_getter = getattr(module, "get_act_quantizer", None)

        if callable(wq_getter):
            wq = wq_getter()
            if wq is not None:
                wq.set_n_bit_manually(self.w_bit)
                applied = getattr(wq, "n_bit", getattr(wq, "bit", None))
                assert applied == self.w_bit, (
                    f"[BIT] Weight bit not applied: {applied} != {self.w_bit}"
                )

        if callable(aq_getter):
            aq = aq_getter()
            if aq is not None:
                aq.set_n_bit_manually(self.a_bit)
                applied = getattr(aq, "n_bit", getattr(aq, "bit", None))
                assert applied == self.a_bit, (
                    f"[BIT] Act bit not applied: {applied} != {self.a_bit}"
                )


def build_taq(
    *,
    device: Union[str, torch.device] = "cuda",
    debug: bool = False,
    **kwargs,
) -> TAQ:
    fp_model = RealViformer(**kwargs)
    return TAQ(fp_model, device=device, debug=debug)
