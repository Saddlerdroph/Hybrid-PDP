#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""strawman.py

一个非常简单的 baseline（"strawman"），用于和 ISHM/PT 的 runner 保持一致的
调用格式与返回格式，并**支持三类外部查询接口**（count / sum / max）。

核心思想：
  - Strawman 不做任何截断/过滤，直接把原始 values 全部作为 retained_values。
  - 通过与 ISHM.py 相同的方式，把 downstream query 交给外部模块：
        Count.py / Sum.py / Max.py
    这些模块应当在 strawman.py 同目录下，并暴露：
        downstream_query(retained_values, tau_for_noise, beta)

返回格式（与 ISHM.ishm / PT.pt 对齐）：
  (noisy_result, tau_for_noise, retained_count)

注意：
  - Strawman 没有 "best_tau" 的选择过程；这里返回的 tau_for_noise 仅用于
    downstream 噪声标定。
  - 为了尽量兼容历史版本（downstream 用 1/EPSILON 的噪声尺度），默认把
    tau_for_noise 设为 2*EPSILON（若外部模块采用 Laplace(scale=2/tau) 的风格）。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, List, Tuple


# ====== Global privacy parameter for this baseline ======
EPSILON: float = 2**(-26)


def _normalize_query_name(query: str) -> str:
    q = (query or "count").strip().lower()
    alias = {"c": "count", "cnt": "count", "s": "sum", "m": "max"}
    return alias.get(q, q)


def _load_downstream_module(query: str):
    """与 ISHM.py 一致的外部 query 模块加载逻辑。"""
    q = _normalize_query_name(query)
    module_map = {"count": "Count", "sum": "Sum", "max": "Max"}
    if q not in module_map:
        raise ValueError(f"Unknown query type: {query!r}. Expected one of {sorted(module_map)}")

    # 保证无论从哪里运行，Count.py/Sum.py/Max.py 都能被 import。
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
            f"Downstream module {module_name}.py must define: downstream_query(retained_values, tau_for_noise, beta)"
        )
    return mod


def downstream_query(retained_values: List[Any], tau_for_noise: float, beta: float | None, query: str = "count") -> float:
    """Dispatch downstream query to Count.py / Sum.py / Max.py based on `query`."""
    mod = _load_downstream_module(query)
    fn = getattr(mod, "downstream_query")
    return float(fn(retained_values, tau_for_noise, beta))


def strawman(values: List[Any], beta: float | None = None, query: str = "count") -> Tuple[float, float, int]:
    """Strawman entrypoint.

    Args:
        values: 原始 record_value 列表。
        beta: 失败概率（strawman 不使用该值做截断/过滤，但会透传给下游模块）。
        query: 下游查询类型：count / sum / max。

    Returns:
        noisy_result: 下游查询返回的带噪结果
        tau_for_noise: 传给下游的噪声标定参数
        retained_count: retained_values 的数量（strawman 不做截断，因此等于 N）
    """
    # beta 由 runner 传入，并原样透传给下游模块的 downstream_query。

    retained_values = list(values)
    retained_count = int(len(retained_values))

    # 兼容旧版：如果 downstream 采用 Laplace(scale=2/tau)，则 tau=2*EPSILON -> scale=1/EPSILON。
    tau_for_noise = float(EPSILON)
    noisy_result = downstream_query(retained_values, tau_for_noise, beta, query=query)
    return float(noisy_result), float(tau_for_noise), retained_count
