#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# Type aliases
# users: uid -> (user_budget, ordered record_ids)
UsersDict = Dict[str, Tuple[float, List[str]]]
# records: rid -> (record_value, record_budget)
RecordsDict = Dict[str, Tuple[Any, float]]
# full input structure: (users, records)
Structure = Tuple[UsersDict, RecordsDict]


# Lower bound for threshold estimation
LB: float = 2**(-26)


def _normalize_query_name(query: str) -> str:
    # Normalize query name and map common aliases
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
            f"Expected a file named {module_name}.py next to PT.py."
        ) from e

    # Contract: downstream module must expose downstream_query(...)
    if not hasattr(mod, "downstream_query"):
        raise AttributeError(
            f"Downstream module {module_name}.py must define: downstream_query(retained_values, tau_for_noise, beta)"
        )
    return mod


def downstream_query(
    retained_values: List[Any],
    tau_for_noise: float,
    beta: float,
    query: str = "count",
) -> float:
    # Dispatch downstream query to Count.py / Sum.py / Max.py based on `query`
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def prefix_truncation(structure: Structure) -> List[str]:
    # Per-user prefix selection by record budgets under user budget constraint
    users, records = structure
    retained_records: List[str] = []

    for _uid, (user_budget, user_record_ids) in users.items():
        # Skip users with non-positive budget
        ub = float(user_budget) if user_budget is not None else 0.0
        if ub <= 0.0:
            continue

        # Collect (record_budget, record_id) for this user
        items: List[Tuple[float, str]] = []
        for rid in user_record_ids:
            rec = records.get(rid)
            if rec is None:
                continue
            _val, rb = rec
            b = float(rb)
            # Defensive: clamp negative budgets to 0
            if b < 0.0:
                b = 0.0
            items.append((b, rid))

        # Sort by record budget ascending for maximal prefix
        items.sort(key=lambda t: t[0])

        # Take maximal prefix whose sum of budgets stays within user budget
        s = 0.0
        for b, rid in items:
            if s + b <= ub + 1e-12:  # allow tiny floating tolerance
                retained_records.append(rid)
                s += b
            else:
                # Maximal prefix: once we fail, stop for this user
                break

    return retained_records


def _compute_ell(budgets: List[float], lb: float) -> int:
    # Compute number of dyadic intervals needed to cover [lb, max_budget]
    if lb <= 0:
        raise ValueError("lb must be > 0")
    if not budgets:
        return 1
    m = max(budgets)
    if m <= lb:
        return 1
    # ell = ceil(log2(m/lb)) so that lb * 2^ell >= m
    return max(1, int(math.ceil(math.log(m / lb, 2.0))))


def record_threshold_estimation(
    retained_records: List[str],
    structure: Structure,
    beta: float,
    lb: float = LB,
) -> float:
    """
    # Estimate a record-budget threshold using dyadic bins and noisy counts.
    """
    beta = float(beta)
    beta = min(max(beta, 1e-12), 0.999999999999)

    _users, records = structure

    # Gather record budgets for retained records
    budgets: List[float] = []
    for rid in retained_records:
        rec = records.get(rid)
        if rec is None:
            continue
        _val, rb = rec
        b = float(rb)
        if b < 0.0:
            b = 0.0
        budgets.append(b)

    # Number of dyadic intervals to test
    ell = _compute_ell(budgets, lb=lb)

    x = float(lb)
    last_x = x

    for _i in range(ell):
        # Count budgets in [x, 2x) (last interval uses [x, 2x])
        upper = 2.0 * x
        c = 0
        is_last = (_i == ell - 1)
        for b in budgets:
            if is_last:
                if x <= b <= upper:
                    c += 1
            else:
                if x <= b < upper:
                    c += 1

        # Add Laplace noise with scale 2/x
        noise = float(np.random.laplace(loc=0.0, scale=(2.0 / x)))
        noisy_count = float(c) + noise

        # Threshold test: (2 ln(ell/beta)) / x
        rhs = (2.0 * math.log(float(ell) / beta)) / x

        # Return first x that passes the noisy test
        if noisy_count > rhs:
            return x

        last_x = x
        x *= 2.0

    # If none passes, return the last tested x
    return last_x


def pt(structure: Structure, beta: float, query: str = "count") -> Tuple[float, float, int]:
    # Run prefix truncation, estimate threshold, then run downstream query with noise
    retained_records = prefix_truncation(structure)

    beta = float(beta)
    beta = min(max(beta, 1e-12), 0.999999999999)
    beta_half = beta / 2.0

    # Split beta between threshold estimation and downstream query
    threshold = record_threshold_estimation(retained_records, structure, beta=beta_half, lb=LB)

    # Filter retained records by the estimated threshold
    _users, records = structure
    final_records: List[str] = []
    for rid in retained_records:
        rec = records.get(rid)
        if rec is None:
            continue
        _val, rb = rec
        if float(rb) >= float(threshold):
            final_records.append(rid)

    # Map final record ids to values
    retained_values: List[Any] = []
    for rid in final_records:
        rec = records.get(rid)
        if rec is None:
            continue
        v, _rb = rec
        retained_values.append(v)

    # Run downstream query (noise calibrated with threshold/2 and beta/2)
    noisy_result = downstream_query(retained_values, float(threshold) / 2.0, beta_half, query=query)
    return float(noisy_result), float(threshold), len(retained_values)
