#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Any, List

import numpy as np


U_GLOBAL: int = 2**30 + 1


def downstream_query(retained_dataset: List[Any], epsilon: float, beta: float) -> int:
    """Equivalent sampling without allocating U_GLOBAL-sized arrays."""
    epsilon = float(epsilon)
    beta = float(beta)

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0, 1)")

    # Validate numeric inputs
    nums: List[float] = []
    bad: List[Any] = []
    for v in retained_dataset:
        if isinstance(v, (int, float)):
            nums.append(float(v))
        else:
            bad.append(v)

    if bad:
        sample = ", ".join(repr(x) for x in bad[:5])
        raise ValueError(
            f"Max query expects numeric retained_dataset, but found {len(bad)} non-numeric values. "
            f"Examples: {sample}"
        )

    m = len(nums)
    if m == 0:
        return 0

    # Compute the original term
    term = int((2 * math.log2(U_GLOBAL / beta)) / epsilon)
    if term >= len(retained_dataset):
        return -1

    # Convert values to breakpoints
    arr = np.asarray(nums, dtype=float)
    b = np.ceil(arr).astype(np.int64)
    b = np.clip(b, 0, U_GLOBAL)
    b = b[b < U_GLOBAL]

    # Count breakpoint multiplicities
    if b.size > 0:
        uniq, cnts = np.unique(b, return_counts=True)
    else:
        uniq = np.empty(0, dtype=np.int64)
        cnts = np.empty(0, dtype=np.int64)

    # Build segment start points
    if uniq.size == 0 or uniq[0] != 0:
        starts = np.concatenate(([0], uniq))
    else:
        starts = uniq
    starts = np.unique(starts)

    # Build constant-weight segments
    seg_start: List[int] = []
    seg_len: List[int] = []
    seg_logw: List[float] = []

    excluded = 0
    p = 0

    for j, s in enumerate(starts):
        s_int = int(s)

        # Apply dropouts at this breakpoint
        if p < uniq.size and int(uniq[p]) == s_int:
            excluded += int(cnts[p])
            p += 1

        nxt = int(starts[j + 1]) if (j + 1) < starts.size else int(U_GLOBAL)
        L = nxt - s_int
        if L <= 0:
            continue

        count_gt = m - excluded
        S = abs(count_gt - term)
        logw = (-epsilon * float(S)) / 2.0

        seg_start.append(s_int)
        seg_len.append(L)
        seg_logw.append(logw)

    if not seg_start:
        return int(np.random.randint(U_GLOBAL))

    seg_start_arr = np.asarray(seg_start, dtype=np.int64)
    seg_len_arr = np.asarray(seg_len, dtype=np.int64)
    seg_logw_arr = np.asarray(seg_logw, dtype=float)

    # Normalize weights stably
    max_logw = float(np.max(seg_logw_arr))
    w_each = np.exp(seg_logw_arr - max_logw)
    mass = w_each * seg_len_arr.astype(float)

    total = float(np.sum(mass))
    if (not np.isfinite(total)) or total <= 0.0:
        return int(np.random.randint(U_GLOBAL))

    # Sample a segment by mass
    u = float(np.random.random()) * total
    cdf = np.cumsum(mass)
    seg_idx = int(np.searchsorted(cdf, u, side="right"))
    if seg_idx >= cdf.size:
        seg_idx = int(cdf.size - 1)

    prev_cum = 0.0 if seg_idx == 0 else float(cdf[seg_idx - 1])

    # Sample within the segment
    offset = (u - prev_cum) / float(w_each[seg_idx])
    k = int(offset)
    L = int(seg_len_arr[seg_idx])
    if k >= L:
        k = L - 1

    return int(seg_start_arr[seg_idx] + k)
