#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GaussianParams:
    mean: float
    std: float
    lower: float = 1.0
    upper: float = 1_000_000.0


@dataclass
class PoissonCountParams:
    """Per-user record count ~ Poisson(lam), with lower/upper bounds."""
    lam: float
    lower: int = 1
    upper: int = 2**16
    max_tries: int = 1_000_000


def reroll_truncated_normal_int(
    rng: np.random.Generator, gp: GaussianParams, max_tries: int = 1_000_000
) -> int:
    """Sample int(round(N(mean,std))) until it falls in [lower, upper]."""
    if gp.std <= 0:
        raise ValueError("std must be > 0")
    if gp.lower > gp.upper:
        raise ValueError("lower > upper")

    lo = int(np.ceil(gp.lower))
    hi = int(np.floor(gp.upper))
    if lo > hi:
        raise ValueError("rounded lower > rounded upper")

    for _ in range(max_tries):
        k = int(np.round(rng.normal(gp.mean, gp.std)))
        if lo <= k <= hi:
            return k
    raise RuntimeError("too many Gaussian rerolls (check mean/std/bounds)")


def sample_poisson_count_int(rng: np.random.Generator, cp: PoissonCountParams) -> int:
    """Sample Poisson(lam) until >= lower; clamp to upper."""
    if cp.lam <= 0:
        raise ValueError("lamda must be > 0")
    if cp.lower < 0:
        raise ValueError("lower must be >= 0")
    if cp.lower > cp.upper:
        raise ValueError("lower > upper")

    for _ in range(cp.max_tries):
        k = int(rng.poisson(cp.lam))
        if k < cp.lower:
            continue
        return min(k, cp.upper)

    raise RuntimeError("too many Poisson samples (lamda too small for lower?)")


def generate_user_counts_to_cover_n(
    rng: np.random.Generator, n: int, cp: PoissonCountParams
) -> Tuple[np.ndarray, int]:
    """Generate per-user counts whose sum reaches n (last one may be truncated)."""
    counts = []
    total = 0
    while total < n:
        k = sample_poisson_count_int(rng, cp)
        k = min(k, n - total)
        counts.append(k)
        total += k
    return np.array(counts, dtype=int), len(counts)


def write_csv(out_path: str, rows) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description="Generate CSV rows: record_id,user_id,value"
    )
    p.add_argument("--n", type=int, required=True, help="total records")

    # value ~ truncated Gaussian
    p.add_argument("--mean", type=float, default=2**20, help="value mean")
    p.add_argument("--std", type=float, default=2**20, help="value std (>0)")
    p.add_argument("--check", type=float, default=0.0, help="value lower bound")
    p.add_argument("--hat", type=float, default=2**30, help="value upper bound")

    # per-user count ~ Poisson
    p.add_argument("--lamda", type=float, default=16.0, help="Poisson mean (>0)")
    p.add_argument("--min_count", type=int, default=1, help="min per user")
    p.add_argument("--max_count", type=int, default=2**16, help="max per user")

    p.add_argument("--out", type=str, default="data.csv", help="output path")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--no_header", action="store_true", help="(kept for compatibility)")
    args = p.parse_args()

    if args.n <= 0:
        raise ValueError("n must be > 0")
    if args.max_count <= 0:
        raise ValueError("max_count must be > 0")

    rng = np.random.default_rng(args.seed)
    gp = GaussianParams(mean=args.mean, std=args.std, lower=args.check, upper=args.hat)
    cp = PoissonCountParams(lam=args.lamda, lower=args.min_count, upper=args.max_count)

    counts, num_users = generate_user_counts_to_cover_n(rng, args.n, cp)

    rows = []
    record_id = 1
    user_id = 1
    for k in counts:
        for _ in range(k):
            rows.append([record_id, user_id, reroll_truncated_normal_int(rng, gp)])
            record_id += 1
        user_id += 1

    write_csv(args.out, rows)
    print(f"Wrote {args.n} records for {num_users} users -> {args.out}")


if __name__ == "__main__":
    main()
