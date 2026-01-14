#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import random
from typing import List, Optional, Tuple


def read_rows_and_col3_values_strict(path: str) -> Tuple[int, List[float]]:
    col3: List[float] = []
    idx = 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row or all(c.strip() == "" for c in row):
                continue
            idx += 1
            if len(row) < 3:
                raise ValueError(f"Row {idx} has fewer than 3 columns: {row}")
            try:
                col3.append(float(row[2]))
            except ValueError as e:
                raise ValueError(f"Row {idx} column 3 is not a number: {row}") from e
    return len(col3), col3


def sample_gaussian_noises(mean: float, std: float, n: int, seed: Optional[int] = None) -> List[float]:
    if n <= 0:
        return []
    if std < 0:
        raise ValueError("std must be >= 0")
    rng = random.Random(seed) if seed is not None else random
    if std == 0:
        return [mean] * n
    return [rng.gauss(mean, std) for _ in range(n)]


def apply_type_transform(base_values: List[float], col3: List[float], typ: str, scale: float) -> List[float]:
    if typ == "normal":
        return base_values[:]

    if len(col3) != len(base_values):
        raise ValueError("Internal error: col3 length mismatch.")

    scale_factor = math.pow(2.0, scale)
    out: List[float] = []

    for i, v in enumerate(base_values):
        w = col3[i]
        if w <= 0:
            w = 1.0
        if typ == "linear":
            out.append(v * w * scale_factor)
        elif typ == "sqrt":
            out.append(v * math.sqrt(w) * scale_factor)
        else:
            raise ValueError(f"Unknown type: {typ}")

    return out


def clip_values(values: List[float], check: float, hat: float) -> Tuple[List[float], float, float]:
    lo_exp, hi_exp = (check, hat) if check <= hat else (hat, check)
    lo = math.pow(2.0, lo_exp)
    hi = math.pow(2.0, hi_exp)
    return [min(max(v, lo), hi) for v in values], lo, hi


def write_csv(path: str, values: List[float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i, val in enumerate(values, start=1):
            w.writerow([i, val])


def report_stats(noises: List[float], pre: List[float], post: List[float], lo: float, hi: float) -> None:
    if not noises:
        print("No samples generated.")
        return
    print("=== Noise stats ===")
    print(f"count      : {len(noises)}")
    print(f"noise_min  : {min(noises)}")
    print(f"noise_max  : {max(noises)}")
    print("=== Value stats BEFORE clip ===")
    print(f"value_min  : {min(pre)}")
    print(f"value_max  : {max(pre)}")
    print("=== Global bound ===")
    print(f"bound_min  : {lo}")
    print(f"bound_max  : {hi}")
    print("=== Value stats AFTER clip ===")
    print(f"value_min  : {min(post)}")
    print(f"value_max  : {max(post)}")


def main():
    p = argparse.ArgumentParser(
        description="Gaussian -> 2^noise -> (optional) scale by col3 -> clip to [2^check,2^hat] -> CSV id,value"
    )
    p.add_argument("--mean", type=float, default=0.0)
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--check", type=float, default=-20)
    p.add_argument("--hat", type=float, default=4)
    p.add_argument("--type", type=str, default="normal", choices=["normal", "linear", "sqrt"])
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("-i", "--in", dest="infile", type=str, required=True)
    p.add_argument("-o", "--out", type=str, default="output.csv")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    n, col3 = read_rows_and_col3_values_strict(args.infile)
    print(f"Input rows (data) : {n}")

    noises = sample_gaussian_noises(args.mean, args.std, n, args.seed)
    base_values = [math.pow(2.0, z) for z in noises]
    values_preclip = apply_type_transform(base_values, col3, args.type, args.scale)
    values_postclip, lo, hi = clip_values(values_preclip, args.check, args.hat)

    report_stats(noises, values_preclip, values_postclip, lo, hi)
    write_csv(args.out, values_postclip)
    print(f"Saved {len(values_postclip)} rows to {args.out}")


if __name__ == "__main__":
    main()
