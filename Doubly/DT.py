#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ====== Global tau bounds ======
# Minimum tau (avoid near-zero / numerical issues)
TAU_LOWER: float = 2**(-26)
# Maximum tau (upper search / constraint bound)
TAU_UPPER: float = 16


# Type aliases
# users: uid -> (user_budget, ordered record_ids)
UsersDict = Dict[str, Tuple[float, List[str]]]
# records: rid -> (record_value, record_budget)
RecordsDict = Dict[str, Tuple[Any, float]]
# full input structure: (users, records)
Structure = Tuple[UsersDict, RecordsDict]


def doubly_truncation_compute(structure: Structure, tau: float) -> int:
    """
    # Count how many records survive doubly-truncation under a given tau.
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


def doubly_truncation_final(structure: Structure, best_tau: float) -> List[Any]:
    """
    # Return retained record_value list under the chosen tau.
    """
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
            f"Expected a file named {module_name}.py next to ISHM.py."
        ) from e

    # Contract: downstream module must expose downstream_query(...)
    if not hasattr(mod, "downstream_query"):
        raise AttributeError(
            f"Downstream module {module_name}.py must define: downstream_query(retained_values, tau_for_noise)"
        )
    return mod


def downstream_query(retained_values: List[Any], tau_for_noise: float, beta: float, query: str = "count") -> float:
    # Dispatch to Count.py / Sum.py / Max.py based on `query`
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def dt(structure: Structure, beta: float, tau: float, query: str = "count") -> Tuple[float, float, int]:
    # Run doubly-truncation, then apply downstream noisy query
    retained = doubly_truncation_final(structure, tau)
    noisy_count = downstream_query(retained, tau, beta, query=query)
    # Return (noisy result, tau used, retained size)
    return noisy_count, tau, len(retained)
