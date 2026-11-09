"""Probability calibration metrics (substep 3.1c). Library code: numpy only, no simulator imports, so the same
functions apply to recorded data in Step 5 (pairs of predicted top-cell probability and whether that cell was right).

Bins are equal width on [0, 1]: bin k holds lo_k <= p < hi_k, the last bin also holds p = 1.
ECE = sum over non-empty bins of (n_bin / n_total) * |mean predicted - observed fraction correct|; n_total counts only
the samples considered (see p_min). Over-confidence of a bin = mean predicted - observed fraction (positive = the
probabilities claim more than is delivered).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BinRow:
    lo: float
    hi: float
    count: int
    mean_pred: float  # NaN when empty
    frac_correct: float  # NaN when empty

    @property
    def over_confidence(self) -> float:
        return self.mean_pred - self.frac_correct if self.count else math.nan


def _arrays(pred: Sequence[float], correct: Sequence[bool]):
    p = np.asarray(pred, dtype=float).reshape(-1)
    c = np.asarray(correct, dtype=bool).reshape(-1)
    if p.shape != c.shape:
        raise ValueError("pred and correct must have the same length")
    if p.size and (not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0):
        raise ValueError("predicted probabilities must be finite and in [0, 1]")
    return p, c


def reliability_table(pred: Sequence[float], correct: Sequence[bool], bins: int = 10) -> List[BinRow]:
    """One BinRow per equal-width bin (empty bins included with NaN statistics)."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    p, c = _arrays(pred, correct)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for k in range(bins):
        sel = idx == k
        n = int(sel.sum())
        rows.append(BinRow(k / bins, (k + 1) / bins, n, float(p[sel].mean()) if n else math.nan,
                           float(c[sel].mean()) if n else math.nan))
    return rows


def ece(pred: Sequence[float], correct: Sequence[bool], bins: int = 10, p_min: Optional[float] = None) -> float:
    """Expected calibration error. p_min restricts the samples to pred >= p_min (n_total = those samples).
    Returns NaN if no sample is considered."""
    p, c = _arrays(pred, correct)
    if p_min is not None:
        keep = p >= p_min
        p, c = p[keep], c[keep]
    if not p.size:
        return math.nan
    rows = reliability_table(p, c, bins)
    return float(sum(r.count / p.size * abs(r.mean_pred - r.frac_correct) for r in rows if r.count))


def over_confidence_flags(pred: Sequence[float], correct: Sequence[bool], bins: int = 10, min_count: int = 20,
                          max_over: float = 0.10) -> List[BinRow]:
    """Bins with at least min_count samples whose over-confidence exceeds max_over."""
    return [r for r in reliability_table(pred, correct, bins) if r.count >= min_count and r.over_confidence > max_over]


def worst_over_confidence(pred: Sequence[float], correct: Sequence[bool], bins: int = 10,
                          min_count: int = 20) -> float:
    """Largest over-confidence among bins with at least min_count samples (NaN if none qualifies)."""
    vals = [r.over_confidence for r in reliability_table(pred, correct, bins) if r.count >= min_count]
    return max(vals) if vals else math.nan


def format_reliability(rows: Sequence[BinRow]) -> str:
    lines = [f"{'bin':>11}{'n':>7}{'pred':>7}{'obs':>7}{'over':>7}"]
    for r in rows:
        if r.count:
            lines.append(f"{r.lo:>5.1f}-{r.hi:<5.1f}{r.count:>7}{r.mean_pred:>7.3f}{r.frac_correct:>7.3f}"
                         f"{r.over_confidence:>+7.3f}")
        else:
            lines.append(f"{r.lo:>5.1f}-{r.hi:<5.1f}{0:>7}{'-':>7}{'-':>7}{'-':>7}")
    return "\n".join(lines)
