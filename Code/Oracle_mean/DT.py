#!/usr/bin/env python3

from __future__ import annotations

import math
import importlib
import sys
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# Search bounds.
TAU_LOWER: float = 2**(-26)
TAU_UPPER: float = 16
ADP_LOWER: float = 2**(-10)
ADP_UPPER: float = 1

# Input: users map to (budget, ordered record IDs); records map to (value, budget).
UsersDict = Dict[str, Tuple[float, List[str]]]
RecordsDict = Dict[str, Tuple[Any, float]]
Structure = Tuple[UsersDict, RecordsDict]


def doubly_truncation_final(structure: Structure, best_tau: float) -> List[Any]:
    """Return values retained at ``best_tau``."""
    users, records = structure
    tau = float(best_tau)
    if tau <= 0:
        raise ValueError("best_tau must be > 0")

    retained: List[Any] = []

    for _uid, (user_budget, user_record_ids) in users.items():
        maxcount = int(math.floor(float(user_budget) / tau)) if user_budget is not None else 0
        if maxcount <= 0:
            continue

        c = 0
        for rid in user_record_ids:
            if c >= maxcount:
                break
            rec = records.get(rid)
            if rec is None:
                continue
            record_value, record_budget = rec
            if float(record_budget) >= tau:
                retained.append(record_value)
                c += 1

    return retained


def _scale_privacy_budgets(structure: Structure, scale: float) -> Structure:
    """Return a copy of ``structure`` with all privacy budgets scaled."""
    users, records = structure
    scaled_users: UsersDict = {
        uid: (float(user_budget) * scale, list(record_ids))
        for uid, (user_budget, record_ids) in users.items()
    }
    scaled_records: RecordsDict = {
        rid: (record_value, float(record_budget) * scale)
        for rid, (record_value, record_budget) in records.items()
    }
    return scaled_users, scaled_records


def _sample_scaled_taus(num_samples: int = 10, scale: float = 0.1) -> List[float]:
    """Sample taus log-uniformly, then scale each one down."""
    log_taus = np.random.uniform(math.log(ADP_LOWER), math.log(ADP_UPPER), size=num_samples)
    return [float(math.exp(log_tau) * scale) for log_tau in log_taus]


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
            f"Expected a file named {module_name}.py next to ISHM.py."
        ) from e

    if not hasattr(mod, "downstream_query"):
        raise AttributeError(
            f"Downstream module {module_name}.py must define: downstream_query(retained_values, tau_for_noise)"
        )
    return mod


def downstream_query(retained_values: List[Any], tau_for_noise: float, beta: float, query: str = "count") -> float:
    """Run the selected downstream query."""
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def dt(structure: Structure, beta: float, query: str = "count") -> Tuple[float, float, int]:

    scaled_structure = _scale_privacy_budgets(structure, scale=0.1)
    scaled_taus = _sample_scaled_taus()

    trial_results: List[float] = []
    retained_counts: List[int] = []
    for tau in scaled_taus:
        retained = doubly_truncation_final(scaled_structure, tau)
        trial_results.append(downstream_query(retained, tau, beta, query=query))
        retained_counts.append(len(retained))

    middle_results = sorted(trial_results)[2:-2]
    return (
        float(statistics.mean(middle_results)),
        float(statistics.mean(scaled_taus)),
        int(round(statistics.mean(retained_counts))),
    )
