"""Histogram-based MSE calibration used by TAQ (paper Sec. 3.2, Algorithm 1).

* Weight bounds: grid search minimising the MSE between a weight tensor and its
  fake-quantised counterpart (Algorithm 1, line 3).
* Activation bounds: activations are percentile-clipped, accumulated into a
  streaming histogram of K bins, and the bounds minimise the bin-weighted
  reconstruction error J(l, u) of Eq. (2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

EPS = 1e-6
DEFAULT_BINS = 1024
DEFAULT_SEARCH_STEPS = 100


# ---------------------------------------------------------------------------
# Weight bounds: MSE grid search directly on the tensor
# ---------------------------------------------------------------------------
def _quantise_restore(tensor: np.ndarray, lb: float, ub: float, bit: int) -> np.ndarray:
    tensor = np.clip(tensor, lb, ub)
    levels = (1 << bit) - 1
    scale = (ub - lb) / max(levels, 1)
    if scale < EPS:
        return np.full_like(tensor, lb, dtype=np.float64)
    q = np.round((tensor - lb) / scale)
    return q * scale + lb


def _mse(input_fp: np.ndarray, lb: float, ub: float, bit: int) -> float:
    restored = _quantise_restore(input_fp, lb, ub, bit)
    diff = input_fp - restored
    return float(np.mean(diff * diff))


def search_weight_bounds(
    tensor: np.ndarray,
    bit: int,
    *,
    steps: int = DEFAULT_SEARCH_STEPS,
) -> Tuple[float, float]:
    """Shrink (min, max) symmetrically and keep the pair with the lowest MSE."""
    vmin = float(tensor.min())
    vmax = float(tensor.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + EPS

    best_lb, best_ub = vmin, vmax
    best_err = _mse(tensor, best_lb, best_ub, bit)

    diff = (vmax - vmin) / max(2 * steps, 1)
    for i in range(steps + 1):
        lb = vmin + diff * i
        ub = vmax - diff * i
        if ub <= lb:
            break
        err = _mse(tensor, lb, ub, bit)
        if err < best_err:
            best_lb, best_ub, best_err = lb, ub, err

    return best_lb, best_ub


# ---------------------------------------------------------------------------
# Activation bounds: streaming histogram + histogram MSE search
# ---------------------------------------------------------------------------
def _cdf_eval(
    x: float,
    counts: np.ndarray,
    edges: np.ndarray,
    cum_counts: np.ndarray,
    total: float,
) -> float:
    if total <= 0.0:
        return 0.0
    if x <= edges[0]:
        return 0.0
    if x >= edges[-1]:
        return total

    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(counts) - 1)
    left = edges[idx]
    right = edges[idx + 1]
    span = right - left
    if span <= 0.0:
        return float(cum_counts[idx])
    frac = (x - left) / span
    return float(cum_counts[idx] + counts[idx] * frac)


def _rebin_histogram(
    counts: np.ndarray,
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
    num_bins: int,
) -> np.ndarray:
    """Redistribute an existing histogram onto a wider range (linear within bins)."""
    if counts.sum() == 0:
        return np.zeros(num_bins, dtype=np.float64)

    old_edges = np.linspace(old_min, old_max, len(counts) + 1, dtype=np.float64)
    new_edges = np.linspace(new_min, new_max, num_bins + 1, dtype=np.float64)
    densities = counts.astype(np.float64)
    cum_counts = np.concatenate(([0.0], np.cumsum(densities)))
    total = float(cum_counts[-1])

    rebinned = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        a = new_edges[i]
        b = new_edges[i + 1]
        if b <= old_min:
            continue
        if a >= old_max:
            break
        left = max(a, old_min)
        right = min(b, old_max)
        if right <= left:
            continue
        cdf_left = _cdf_eval(left, densities, old_edges, cum_counts, total)
        cdf_right = _cdf_eval(right, densities, old_edges, cum_counts, total)
        rebinned[i] += cdf_right - cdf_left

    return rebinned


def _quant_error(
    centers: np.ndarray,
    weights: np.ndarray,
    lb: float,
    ub: float,
    bit: int,
) -> float:
    """J(l, u) of Eq. (2): count-weighted squared error of the bin centres."""
    if ub <= lb:
        return math.inf

    levels = (1 << bit) - 1
    span = ub - lb
    if span <= 0:
        return 0.0
    step = span / levels
    if step == 0:
        return 0.0

    clipped = np.clip(centers, lb, ub)
    q = np.round((clipped - lb) / step) * step + lb
    diff = centers - q
    return float(np.sum(weights * diff * diff))


def _histogram_mse_search(
    centers: np.ndarray,
    weights: np.ndarray,
    bit: int,
    num_steps: int = DEFAULT_SEARCH_STEPS,
) -> Tuple[float, float]:
    mask = weights > 0
    if not np.any(mask):
        raise ValueError("Histogram weights are all zero.")

    vals = centers[mask]
    wts = weights[mask]
    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + EPS

    best_lb = vmin
    best_ub = vmax
    best_err = math.inf

    diff = (vmax - vmin) / max(2 * num_steps, 1)
    for i in range(num_steps + 1):
        lb = vmin + diff * i
        ub = vmax - diff * i
        if ub <= lb:
            break
        err = _quant_error(vals, wts, lb, ub, bit)
        if err < best_err:
            best_lb, best_ub, best_err = lb, ub, err

    return best_lb, best_ub


@dataclass
class StreamingHistogramObserver:
    """Maintain a K-bin histogram over a stream of tensors, widening the range as needed."""

    num_bins: int = DEFAULT_BINS
    hist: Optional[np.ndarray] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    total_count: int = 0

    def update(self, tensor: Union[np.ndarray, torch.Tensor, float]) -> None:
        arr = np.asarray(tensor, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        current_min = float(arr.min())
        current_max = float(arr.max())
        if np.isclose(current_min, current_max):
            current_max = current_min + EPS

        if self.hist is None:
            self.min_val = current_min
            self.max_val = current_max
            self.hist, _ = np.histogram(
                arr,
                bins=self.num_bins,
                range=(self.min_val, self.max_val),
            )
        else:
            new_min = min(self.min_val, current_min)
            new_max = max(self.max_val, current_max)
            if not np.isclose(new_min, self.min_val) or not np.isclose(new_max, self.max_val):
                self.hist = _rebin_histogram(
                    self.hist,
                    self.min_val,
                    self.max_val,
                    new_min,
                    new_max,
                    self.num_bins,
                )
                self.min_val = new_min
                self.max_val = new_max

            counts, _ = np.histogram(
                arr,
                bins=self.num_bins,
                range=(self.min_val, self.max_val),
            )
            self.hist += counts

        self.total_count += arr.size

    def compute_bounds(self, *, bit: int, num_steps: int = DEFAULT_SEARCH_STEPS) -> Tuple[float, float]:
        if self.hist is None or self.total_count == 0:
            raise ValueError("Observer has no data.")

        edges = np.linspace(self.min_val, self.max_val, self.num_bins + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = self.hist.astype(np.float64)
        return _histogram_mse_search(centers, weights, bit, num_steps)


class MSEQuantCalibrator:
    """Percentile-clip incoming activations, accumulate a histogram, and solve Eq. (2)."""

    def __init__(
        self,
        *,
        bit: int,
        bins: int = DEFAULT_BINS,
        clip_quantile_low: Optional[float] = 0.001,
        clip_quantile_high: Optional[float] = 0.999,
    ) -> None:
        self.bit = bit
        self.observer = StreamingHistogramObserver(num_bins=bins)
        if (clip_quantile_low is None) != (clip_quantile_high is None):
            raise ValueError("clip_quantile_low and clip_quantile_high must be set together.")
        if clip_quantile_low is not None and clip_quantile_high is not None:
            if not (0.0 <= clip_quantile_low < clip_quantile_high <= 1.0):
                raise ValueError("clip quantiles must satisfy 0 <= low < high <= 1.")
        self.clip_quantile_low = clip_quantile_low
        self.clip_quantile_high = clip_quantile_high

    def observe(self, tensor: torch.Tensor) -> None:
        arr = tensor.detach().to(dtype=torch.float32).cpu().numpy()
        if self.clip_quantile_low is not None and self.clip_quantile_high is not None:
            lo_b, hi_b = np.quantile(arr, [self.clip_quantile_low, self.clip_quantile_high])
            arr = np.clip(arr, lo_b, hi_b)
        self.observer.update(arr)

    def finalise(self, *, steps: int = DEFAULT_SEARCH_STEPS) -> Tuple[float, float]:
        return self.observer.compute_bounds(bit=self.bit, num_steps=steps)
