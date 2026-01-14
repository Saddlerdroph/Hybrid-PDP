#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, List, Tuple


# ====== Global privacy parameter for this baseline ======
EPSILON: float = 2 ** (-26)


def _normalize_query_name(query: str) -> str:
    """Normalize query aliases to canonical names."""
    q = (query or "count").strip().lower()
    alias = {"c": "count", "cnt": "count", "s": "sum", "m": "max"}
    return alias.get(q, q)


def _load_downstream_module(query: str):
    """
    Load the downstream query module.

    This follows the same external query-module loading logic as ISHM.py.
    """
    q = _normalize_query_name(query)
    module_map = {"count": "Count", "sum": "Sum", "max": "Max"}
    if q not in module_map:
        raise ValueError(
            f"Unknown query type: {query!r}. Expected one of {sorted(module_map)}"
        )

    # Ensure Count.py / Sum.py / Max.py can be imported regardless of where this is run.
    this_dir = str((Path(__file__).resolve().parent.parent / "downstream").resolve())
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    module_name = module_map[q]
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import downstream module {module_name!r}. "
            f"Expected a file named {module_name}.py next to strawman.py."
        ) from e

    if not hasattr(mod, "downstream_query"):
        raise AttributeError(
            f"Downstream module {module_name}.py must define: "
            f"downstream_query(retained_values, tau_for_noise, beta)"
        )
    return mod


def downstream_query(
    retained_values: List[Any],
    tau_for_noise: float,
    beta: float | None,
    query: str = "count",
) -> float:
    """
    Dispatch the downstream query to Count.py / Sum.py / Max.py
    based on the given `query` type.
    """
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def strawman(
    values: List[Any],
    beta: float | None = None,
    query: str = "count",
) -> Tuple[float, float, int]:
    """
    Strawman entry point.

    Args:
        values: List of original record values.
        beta: Failure probability (not used for truncation/filtering
              in strawman, but passed through to downstream modules).
        query: Downstream query type: count / sum / max.

    Returns:
        noisy_result: Noisy result returned by the downstream query
        tau_for_noise: Noise calibration parameter passed downstream
        retained_count: Number of retained values (equal to N in strawman)
    """
    # beta is provided by the runner and passed through unchanged
    # to the downstream module's downstream_query.

    retained_values = list(values)
    retained_count = int(len(retained_values))

    # Backward compatibility:
    # If downstream uses Laplace(scale = 2 / tau),
    # then tau = 2 * EPSILON -> scale = 1 / EPSILON.
    tau_for_noise = float(EPSILON)
    noisy_result = downstream_query(
        retained_values, tau_for_noise, beta, query=query
    )
    return float(noisy_result), float(tau_for_noise), retained_count
