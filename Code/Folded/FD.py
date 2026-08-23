#!/usr/bin/env python3

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# Input: users map to (budget, ordered record IDs); records map to (value, budget).
UsersDict = Dict[str, Tuple[float, List[str]]]
RecordsDict = Dict[str, Tuple[Any, float]]
Structure = Tuple[UsersDict, RecordsDict]


# Threshold lower bound.
LB: float = 2**(-26)
M_HAT: int = 2**16


def _normalize_query_name(query: str) -> str:
    # Normalize common aliases.
    q = (query or "count").strip().lower()
    alias = {"c": "count", "cnt": "count", "s": "sum", "m": "max"}
    return alias.get(q, q)


def _load_downstream_module(query: str):
    q = _normalize_query_name(query)
    module_map = {"count": "Count", "sum": "Sum", "max": "Max"}
    if q not in module_map:
        raise ValueError(f"Unknown query type: {query!r}. Expected one of {sorted(module_map)}")

    # Load sibling query modules from ``downstream``.
    this_dir = str((Path(__file__).resolve().parent.parent / "downstream").resolve())
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    module_name = module_map[q]
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import downstream module {module_name!r}. "
            f"Expected a file named {module_name}.py next to FD.py."
        ) from e

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
    """Run the selected downstream query."""
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def prefix_truncation(structure: Structure) -> List[str]:
    """Select each user's maximal budget-feasible record prefix."""
    users, records = structure
    retained_records: List[str] = []

    for _uid, (user_budget, user_record_ids) in users.items():
        ub = float(user_budget) if user_budget is not None else 0.0
        if ub <= 0.0:
            continue

        items: List[Tuple[float, str]] = []
        for rid in user_record_ids:
            rec = records.get(rid)
            if rec is None:
                continue
            _val, rb = rec
            b = float(rb)
            if b < 0.0:
                b = 0.0
            items.append((b, rid))

        items.sort(key=lambda t: t[0])

        s = 0.0
        for b, rid in items:
            if s + b <= ub + 1e-12:
                retained_records.append(rid)
                s += b
            else:
                break

    return retained_records


def fold_record_budgets(structure: Structure, m_hat: int = M_HAT) -> Tuple[Structure, List[str]]:
    """Cap record budgets at each associated user's budget divided by ``m_hat``."""
    if m_hat <= 0:
        raise ValueError("m_hat must be > 0")

    users, records = structure
    folded_records: RecordsDict = dict(records)
    retained_records: List[str] = []
    seen_records = set()

    for _uid, (user_budget, user_record_ids) in users.items():
        cap = (float(user_budget) if user_budget is not None else 0.0) / float(m_hat)
        for rid in user_record_ids:
            rec = folded_records.get(rid)
            if rec is None:
                continue
            record_value, record_budget = rec
            folded_records[rid] = (record_value, min(float(record_budget), cap))
            if rid not in seen_records:
                retained_records.append(rid)
                seen_records.add(rid)

    return (users, folded_records), retained_records


def _compute_ell(budgets: List[float], lb: float) -> int:
    if lb <= 0:
        raise ValueError("lb must be > 0")
    if not budgets:
        return 1
    m = max(budgets)
    if m <= lb:
        return 1
    return max(1, int(math.ceil(math.log(m / lb, 2.0))))


def record_threshold_estimation(
    retained_records: List[str],
    structure: Structure,
    beta: float,
    lb: float = LB,
) -> float:
    """Estimate a record-budget threshold from noisy dyadic-bin counts."""
    beta = float(beta)
    beta = min(max(beta, 1e-12), 0.999999999999)

    _users, records = structure

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

    ell = _compute_ell(budgets, lb=lb)

    x = float(lb)
    last_x = x

    for _i in range(ell):
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

        noise = float(np.random.laplace(loc=0.0, scale=(2.0 / x)))
        noisy_count = float(c) + noise

        rhs = (2.0 * math.log(float(ell) / beta)) / x

        if noisy_count > rhs:
            return x

        last_x = x
        x *= 2.0

    return last_x


def pt(structure: Structure, beta: float, query: str = "count") -> Tuple[float, float, int]:
    """Fold budgets, estimate a threshold, and run the downstream query."""
    folded_structure, retained_records = fold_record_budgets(structure)

    beta = float(beta)
    beta = min(max(beta, 1e-12), 0.999999999999)
    beta_half = beta / 2.0

    threshold = record_threshold_estimation(retained_records, folded_structure, beta=beta_half, lb=LB)

    _users, records = folded_structure
    final_records: List[str] = []
    for rid in retained_records:
        rec = records.get(rid)
        if rec is None:
            continue
        _val, rb = rec
        if float(rb) >= float(threshold):
            final_records.append(rid)

    retained_values: List[Any] = []
    for rid in final_records:
        rec = records.get(rid)
        if rec is None:
            continue
        v, _rb = rec
        retained_values.append(v)

    noisy_result = downstream_query(retained_values, float(threshold) / 2.0, beta_half, query=query)
    return float(noisy_result), float(threshold), len(retained_values)
