import math
import numpy as np
from typing import List


def laplace_noise(scale: float) -> float:
    return float(np.random.laplace(loc=0.0, scale=scale))


def count_leq(D: List[float], x: float) -> int:
    return sum(1 for v in D if v <= x)


def svt(T: float, epsilon: float, queries, D: List[float]) -> int:
    T_tilde = T + laplace_noise(scale=2.0 / epsilon)

    for i, Q_i in enumerate(queries, start=1):
        noisy_val = Q_i(D) + laplace_noise(scale=4.0 / epsilon)
        if noisy_val > T_tilde:
            return i

    return len(queries) + 1


def dp_estimate_max(
    D: List[float],
    epsilon: float,
    beta: float,
    max_exponent: int = 31,
) -> float:
    n = len(D)
    if n == 0:
        return 0.0

    T = n - (6.0 / epsilon) * math.log(2.0 / beta)
    queries = [lambda data: count_leq(data, 0.0)]
    for i in range(2, max_exponent + 2):
        p = 2 ** (i - 2)
        queries.append(lambda data, val=p: count_leq(data, val))

    i_tilde = svt(T, epsilon, queries, D)

    if i_tilde == 1:
        return 0.0
    elif i_tilde > len(queries):
        return float(2 ** (max_exponent - 1))
    else:
        return float(2 ** (i_tilde - 2))


def dp_sum_after_clipping(
    D: List[float],
    clip_value: float,
    epsilon_sum: float,
) -> float:
    clipped_sum = sum(min(x, clip_value) for x in D)
    noise = laplace_noise(scale=clip_value / epsilon_sum)
    return clipped_sum + noise


def downstream_query(
    values: List[float],
    epsilon: float,
    beta: float,
) -> float:
    
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if beta <= 0 or beta >= 1:
        raise ValueError("beta must be in (0, 1)")

    nums: List[float] = []
    bad = []
    for v in values:
        if isinstance(v, (int, float)):
            nums.append(float(v))
        else:
            bad.append(v)

    if bad:
        sample = ", ".join(repr(x) for x in bad[:5])
        raise ValueError(
            f"Sum query expects numeric values, but found {len(bad)} non-numeric values. "
            f"Examples: {sample}"
        )

    if not nums:
        return 0.0

    eps_max = epsilon * 4.0 / 5.0
    eps_sum = epsilon * 1.0 / 5.0

    dp_max = dp_estimate_max(nums, eps_max, beta)
    dp_sum = dp_sum_after_clipping(nums, dp_max, eps_sum)

    return dp_sum
