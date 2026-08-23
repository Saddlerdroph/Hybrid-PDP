#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


# Input: users map to (budget, ordered record IDs); records map to (value, budget).
UsersDict = Dict[str, Tuple[float, List[str]]]
RecordsDict = Dict[str, Tuple[Any, float]]
Structure = Tuple[UsersDict, RecordsDict]

M_HAT: int = 2**16
NUM_TAUS: int = 45
TAU_START_EXPONENT: int = 12


def fold_record_budgets(structure: Structure, m_hat: int = M_HAT) -> Structure:
    """Cap each record budget by its associated user's budget / ``m_hat``."""
    if m_hat <= 0:
        raise ValueError("m_hat must be > 0")

    users, records = structure
    folded_records: RecordsDict = dict(records)

    for _uid, (user_budget, user_record_ids) in users.items():
        cap = (float(user_budget) if user_budget is not None else 0.0) / float(m_hat)
        for rid in user_record_ids:
            rec = folded_records.get(rid)
            if rec is None:
                continue
            record_value, record_budget = rec
            folded_records[rid] = (record_value, min(float(record_budget), cap))

    return users, folded_records


def _tau_values() -> List[float]:
    """Return 45 tau values from 2^12 through 2^56."""
    return [float(2 ** (TAU_START_EXPONENT + i)) for i in range(NUM_TAUS)]


def folded_sum(structure: Structure, beta: float = 0.1) -> float:
    """Evaluate all folded sum candidates and return the largest noisy sum."""
    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")

    _users, records = fold_record_budgets(structure)
    numeric_records: List[Tuple[float, float]] = []

    for rid, (record_value, record_budget) in records.items():
        if not isinstance(record_value, (int, float)):
            raise ValueError(f"Folded-sum requires numeric record values; {rid!r} has {record_value!r}")
        numeric_records.append((float(record_value), float(record_budget)))

    iteration_results: List[float] = []
    for tau in _tau_values():
        clipped_sum = 0.0
        for value, record_budget in numeric_records:
            if record_budget <= 0.0:
                continue
            clipped_sum += min(value, tau * record_budget / NUM_TAUS)
        noise = float(np.random.laplace(loc=0.0, scale=tau))
        suppression = tau * math.log(NUM_TAUS / beta)
        iteration_results.append(clipped_sum + noise - suppression)

    return float(max(iteration_results))


def pt(structure: Structure, beta: float = 0.1) -> float:
    """Compatibility entry point for the Folded-sum baseline."""
    return folded_sum(structure, beta=beta)
