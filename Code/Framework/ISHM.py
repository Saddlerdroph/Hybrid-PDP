#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import importlib
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# ====== Global tau bounds ======
# Minimum tau (avoid near-zero / numerical issues)
TAU_LOWER: float = 2 ** (-26)
# Maximum tau (upper search / constraint bound)
TAU_UPPER: float = 16
 


# Type aliases
# users: uid -> (user_budget, ordered record_ids)
UsersDict = Dict[str, Tuple[float, List[str]]]
# records: rid -> (record_value, record_budget)
RecordsDict = Dict[str, Tuple[Any, float]]
# full input structure: (users, records)
Structure = Tuple[UsersDict, RecordsDict]


def _iter_tau_grid() -> List[float]:
    # Generate tau candidates: TAU_LOWER, 2*TAU_LOWER, ..., up to TAU_UPPER (inclusive, with epsilon)
    taus: List[float] = []
    tau = float(TAU_LOWER)
    upper = float(TAU_UPPER) * (1.0 + 1e-12)
    while tau <= upper:
        taus.append(tau)
        tau *= 2.0
    return taus


def doubly_truncation_compute(structure: Structure, tau: float) -> int:
    """
    # Count records retained by doubly truncation under a given tau (summed across all users).
    """
    users, records = structure
    tau = float(tau)
    # tau must be positive
    if tau <= 0:
        raise ValueError("tau must be > 0")

    total = 0
    for _uid, (user_budget, user_record_ids) in users.items():
        # Per-user cap: floor(user_budget / tau)
        maxcount = int(math.floor(float(user_budget) / tau)) if user_budget is not None else 0
        if maxcount <= 0:
            continue

        c = 0
        for rid in user_record_ids:
            # Stop once reaching per-user cap
            if c >= maxcount:
                break
            # Missing record id -> skip
            rec = records.get(rid)
            if rec is None:
                continue
            _val, record_budget = rec
            # Keep record only if record_budget >= tau
            if float(record_budget) >= tau:
                c += 1

        # Accumulate retained count across users
        total += c

    return total


def _doubly_truncation_counts_one_user(
    user_budget: float,
    user_record_ids: List[str],
    records: RecordsDict,
    taus: List[float],
) -> List[int]:
    # For one user: scan the record list once and return its contribution count[i] for each tau.
    
    K = len(taus)
    if user_budget is None:
        return [0] * K

    B = float(user_budget)

    # maxcount[i] = floor(B / tau_i); non-positive means the tau is saturated from the start
    maxcount = [0] * K
    for i, t in enumerate(taus):
        mc = int(math.floor(B / t))
        maxcount[i] = mc if mc > 0 else 0

    counts = [0] * K

    # DSU / next-pointer: quickly skip tau indices that are already saturated
    parent = list(range(K + 1))  # parent[K] is a sentinel
    for i, mc in enumerate(maxcount):
        if mc <= 0:
            parent[i] = i + 1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid in user_record_ids:
        i0 = find(0)
        if i0 >= K:
            break  # all taus are saturated, equivalent to early stop for every tau

        rec = records.get(rid)
        if rec is None:
            continue
        _val, record_budget = rec
        rb = float(record_budget)

        # This record is eligible for all taus <= rb (taus are sorted ascending)
        hi = bisect_right(taus, rb) - 1
        if hi < 0:
            continue

        i = i0
        while i <= hi:
            counts[i] += 1
            if counts[i] >= maxcount[i]:
                parent[i] = find(i + 1)  # tau i is now saturated; remove it
                i = parent[i]
            else:
                i = find(i + 1)

    return counts


def doubly_truncation_counts_grid(
    structure: Structure,
    taus: List[float],
    return_per_user: bool = False
) -> Tuple[List[int], Optional[Dict[str, List[int]]]]:
    # Scan all users (each user scans its record list once) and compute global count for each tau.

    users, records = structure
    K = len(taus)
    total = [0] * K
    per_user = {} if return_per_user else None

    for uid, (user_budget, user_record_ids) in users.items():
        contrib = _doubly_truncation_counts_one_user(user_budget, user_record_ids, records, taus)
        if return_per_user:
            per_user[uid] = contrib

        for i in range(K):
            total[i] += contrib[i]

    return total, per_user


def threshold_selection(structure: Structure, beta: float) -> float:
    # Serial threshold selection:
    # - build tau grid
    # - scan each user's records once to get counts for all taus
    # - sample noise and compute score in tau order, pick the best
    
    beta = float(beta)
    beta = min(max(beta, 1e-12), 0.999999999999)
    ell = math.log2(TAU_UPPER/TAU_LOWER)+1

    taus = _iter_tau_grid()
    if not taus:
        return float(TAU_LOWER)

    # One pass to get counts for all taus
    counts, _ = doubly_truncation_counts_grid(structure, taus, return_per_user=False)

    best_tau = taus[0]
    maxscore = float("-inf")

    # Sample noise per tau in order (keeps serial behavior)
    for count, tau in zip(counts, taus):
        b = 2.0 * ell / float(tau)
        noise = float(np.random.laplace(loc=0.0, scale=b))
        C = 3.0 * b * math.log(ell / beta)
        score = float(count) + noise - C

        if score >= maxscore:
            maxscore = score
            best_tau = float(tau)

    return float(best_tau)


def doubly_truncation_final(structure: Structure, best_tau: float) -> List[Any]:
    users, records = structure
    tau = float(best_tau)
    # tau must be positive
    if tau <= 0:
        raise ValueError("best_tau must be > 0")

    retained: List[Any] = []

    for _uid, (user_budget, user_record_ids) in users.items():
        # Per-user cap: floor(user_budget / tau)
        maxcount = int(math.floor(float(user_budget) / tau)) if user_budget is not None else 0
        if maxcount <= 0:
            continue

        c = 0
        for rid in user_record_ids:
            # Stop once reaching per-user cap
            if c >= maxcount:
                break
            # Missing record id -> skip
            rec = records.get(rid)
            if rec is None:
                continue
            record_value, record_budget = rec
            # Keep record only if record_budget >= tau
            if float(record_budget) >= tau:
                retained.append(record_value)
                c += 1

    return retained


def _normalize_query_name(query: str) -> str:

    q = (query or "count").strip().lower()
    alias = {"c": "count", "cnt": "count", "s": "sum", "m": "max"}
    return alias.get(q, q)


def _load_downstream_module(query: str):
    # Map query type to sibling module name
    q = _normalize_query_name(query)
    module_map = {"count": "Count", "sum": "Sum", "max": "Max"}
    if q not in module_map:
        raise ValueError(f"Unknown query type: {query!r}. Expected one of {sorted(module_map)}")

    # Ensure downstream/ is importable regardless of current working directory
    this_dir = str((Path(__file__).resolve().parent.parent / "downstream").resolve())
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    module_name = module_map[q]
    try:
        # Import downstream module dynamically
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import downstream module {module_name!r}. "
            f"Expected a file named {module_name}.py next to ISHM.py."
        ) from e

    # Contract: downstream module must expose downstream_query(...)
    if not hasattr(mod, "downstream_query"):
        raise AttributeError(
            f"Downstream module {module_name}.py must define: downstream_query(retained_values, tau_for_noise, beta)"
        )
    return mod


def downstream_query(retained_values: List[Any], tau_for_noise: float, beta: float, query: str = "count") -> float:
    # Dispatch downstream query to Count.py / Sum.py / Max.py based on `query`
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def ishm(structure: Structure, beta: float, query: str = "count") -> Tuple[float, float, int]:
    # Select tau, apply doubly truncation, then run downstream query with noise
    best_tau = threshold_selection(structure, beta / 2.0)
    retained = doubly_truncation_final(structure, best_tau)
    noisy_ans = downstream_query(retained, best_tau / 2.0, beta / 2.0, query=query)
    return noisy_ans, best_tau, len(retained)
