#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import math
import statistics
import time
import csv
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import ISHM


# Candidate filenames expected next to the dataset file
RECORD_MAPPING_CANDIDATES = [
    "record.csv"
]
USER_MAPPING_CANDIDATES = [
    "user.csv"
]


def _collect_numeric_values(structure) -> List[float]:
    # Collect record_value as floats; fail if any value is non-numeric
    _, records = structure
    values: List[float] = []
    for rid, (val, _budget) in records.items():
        if isinstance(val, (int, float)):
            values.append(float(val))
        else:
            raise ValueError(
                f"Non-numeric record_value encountered for record_id={rid!r}: {val!r}. "
                f"sum/max queries require numeric values."
            )
    return values


def _trim_finite(xs: List[float], trim_ratio: float = 0.0) -> List[float]:
    """
    # Return sorted finite floats, with optional symmetric trimming.
    # trim_ratio=0.2 drops the lowest 20% and highest 20%.
    """
    # Keep only finite numbers
    xs2 = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs2:
        return []
    # Sort for trimming
    xs2.sort()
    if trim_ratio <= 0:
        return xs2
    n = len(xs2)
    # Number to drop on each side
    k = int(math.floor(n * trim_ratio))
    # If trimming removes everything, fall back to no trim
    if 2 * k >= n:
        return xs2
    return xs2[k:n - k]


def _finite_mean(xs: List[float], trim_ratio: float = 0.0) -> float:
    # Mean over finite (and optionally trimmed) values
    xs2 = _trim_finite(xs, trim_ratio=trim_ratio)
    return statistics.mean(xs2) if xs2 else float("nan")


def _finite_pstdev(xs: List[float], trim_ratio: float = 0.0) -> float:
    # Population stddev over finite (and optionally trimmed) values
    xs2 = _trim_finite(xs, trim_ratio=trim_ratio)
    return statistics.pstdev(xs2) if len(xs2) > 1 else float("nan")


def _append_csv_row(csv_path: Path, header: List[str], row: List[Any]) -> None:
    # Append one row; create file + header if missing/empty
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def _parse_value(s: str) -> Union[float, str]:
    # Parse numeric value if possible; otherwise keep as string
    try:
        return float(s)
    except Exception:
        return s


def _split_parts(line: str) -> List[str]:
    # Split columns; accept commas and/or whitespace as separators
    return line.replace(",", " ").split()


def _read_mapping_file(path: str, value_name: str) -> Dict[str, float]:
    # Read "id value" mapping file; skip blanks and comments
    mp: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _split_parts(line)
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected 'id {value_name}', got: {line!r}")
            k = parts[0]
            try:
                v = float(parts[1])
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: cannot parse {value_name} as float: {parts[1]!r}") from e
            mp[k] = v
    return mp


def _find_sibling_file(dataset_path: str, candidates: List[str]) -> str:
    # Find a required file in the dataset directory using candidate names
    directory = os.path.dirname(os.path.abspath(dataset_path)) or "."
    for name in candidates:
        p = os.path.join(directory, name)
        if os.path.exists(p) and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find required sibling file in directory: {directory}\n"
        f"Tried names: {candidates}"
    )


def preprocess(dataset_path: str) -> Tuple[
    Tuple[Dict[str, Tuple[float, List[str]]], Dict[str, Tuple[Any, float]]],
    int,
    str,
    str,
]:
    # Resolve mapping files next to dataset
    record_mapping_path = _find_sibling_file(dataset_path, RECORD_MAPPING_CANDIDATES)
    user_mapping_path = _find_sibling_file(dataset_path, USER_MAPPING_CANDIDATES)

    # Load privacy budgets from mapping files
    record_budget_map = _read_mapping_file(record_mapping_path, "record_privacy_budget")
    user_budget_map = _read_mapping_file(user_mapping_path, "user_privacy_budget")

    # Build user -> record_ids and record_id -> value
    user_records: Dict[str, List[str]] = defaultdict(list)
    record_values: Dict[str, Any] = {}

    # Parse dataset lines: record_id user_id record_value
    n_total = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _split_parts(line)
            if len(parts) < 3:
                raise ValueError(
                    f"{dataset_path}:{lineno}: expected 'record_id user_id record_value', got: {line!r}"
                )
            rid, uid, raw_val = parts[0], parts[1], parts[2]
            record_values[rid] = _parse_value(raw_val)
            user_records[uid].append(rid)
            n_total += 1

    # users_dict: uid -> (user_budget, ordered record_id_list)
    users_dict: Dict[str, Tuple[float, List[str]]] = {}
    for uid, rids in user_records.items():
        users_dict[uid] = (float(user_budget_map.get(uid, 0.0)), rids)

    # records_dict: rid -> (record_value, record_budget)
    records_dict: Dict[str, Tuple[Any, float]] = {}
    for rid, val in record_values.items():
        records_dict[rid] = (val, float(record_budget_map.get(rid, 0.0)))

    structure = (users_dict, records_dict)
    return structure, n_total, record_mapping_path, user_mapping_path


def main() -> None:
    # CLI: dataset path + query mode + ISHM params
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file", help="Path to dataset_file (record_id user_id record_value per line)")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="Number of rounds to run ISHM")
    parser.add_argument("-beta", "--beta", type=float, default=0.1, help="Failure rate beta (0<beta<1)")
    parser.add_argument(
        "-q", "--query",
        choices=["count", "sum", "max"],
        default="count",
        help="Downstream query type: count / sum / max"
    )
    args = parser.parse_args()

    # Load and normalize dataset + mappings
    structure, n_total, record_map_path, user_map_path = preprocess(args.dataset_file)

    # Compute ground truth for error metrics
    if args.query == "count":
        truth = float(n_total)
        values_all = None
    else:
        values_all = _collect_numeric_values(structure)
        if args.query == "sum":
            truth = float(sum(values_all))
        elif args.query == "max":
            truth = float(max(values_all)) if values_all else float("-inf")
        else:
            raise ValueError(f"Unknown query: {args.query!r}")

    # Guard: empty dataset after filtering comments/blanks
    if n_total <= 0:
        raise SystemExit("dataset_file contains 0 records (after removing empty/comment lines).")

    # Per-round logs and summary accumulators
    times: List[float] = []
    noisy_results: List[float] = []
    abs_errors: List[float] = []
    rel_errors: List[float] = []
    noisy_maxes: List[float] = []
    rank_errors: List[float] = []
    rank_ratios: List[float] = []

    # Print run configuration
    print("=== Input ===")
    print(f"dataset_file        : {os.path.abspath(args.dataset_file)}")
    print(f"record_mapping_file : {record_map_path}")
    print(f"user_mapping_file   : {user_map_path}")
    print(f"total_records (N)   : {n_total}")
    print(f"rounds              : {args.rounds}")
    print(f"beta                : {args.beta}")
    print(f"query               : {args.query}")
    if args.query == "count":
        print(f"truth_count (N)     : {truth:g}")
    elif args.query == "sum":
        print(f"truth_sum           : {truth:.6f}")
    else:
        print(f"truth_max           : {truth:.6f}")
    print()

    # Run multiple rounds and collect metrics
    for r in range(1, args.rounds + 1):
        # Time the ISHM call only
        t0 = time.perf_counter()
        noisy_result, best_tau, retained_count = ISHM.ishm(structure, beta=args.beta, query=args.query)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times.append(elapsed)

        if args.query in ("count", "sum"):
            # Absolute/relative error vs ground truth
            abs_err = abs(noisy_result - truth)
            denom = abs(truth)
            rel_err = abs_err / denom if denom > 0 else float("nan")
            rel_errors.append(rel_err)
            noisy_results.append(float(noisy_result))
            abs_errors.append(abs_err)

            # Round output line
            label = "noisy_count" if args.query == "count" else "noisy_sum"
            print(
                f"[Round {r:>3}] best_tau={best_tau:g}  retained={retained_count}  "
                f"{label}={noisy_result:.6f}  time={elapsed*1000:.3f} ms  rel_error={rel_err:.6f}"
            )
        else:
            # max: rank error = count(true_value > reported_max)
            if values_all is None:
                raise RuntimeError("values_all should have been computed for max query")
            rank_error = sum(1 for v in values_all if v > noisy_result)
            rank_ratio = rank_error / len(values_all) if values_all else float("nan")

            # Store metrics (keep schema similar to count/sum)
            noisy_results.append(float(noisy_result))
            abs_err = abs(rank_error)
            denom = abs(float(n_total))
            rel_err = abs_err / denom if denom > 0 else float("nan")
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
            noisy_maxes.append(float(noisy_result))
            rank_errors.append(float(rank_error))
            rank_ratios.append(float(rank_ratio))

            # Round output line
            print(
                f"[Round {r:>3}] best_tau={best_tau:g}  retained={retained_count}  "
                f"noisy_max={noisy_result:.6f}   time={elapsed*1000:.3f} ms  "
                f"rank_error={rank_error}  rank_ratio={rank_ratio:.6f}"
            )

    # Summarize with trimmed statistics to reduce outlier impact
    print("\n=== Summary ===")
    trim_ratio = 0.2  # default: keep middle 60%
    print(f"trim_ratio    : {trim_ratio:g}  (drop lowest/highest {trim_ratio*100:.0f}% values)")

    times_trim = _trim_finite(times, trim_ratio=trim_ratio)
    print(f"kept_rounds   : {len(times_trim)}/{len(times)}")

    print(f"avg_time_ms   : {_finite_mean(times, trim_ratio=trim_ratio)*1000:.3f}")
    if len(times_trim) > 1:
        print(f"std_time_ms   : {_finite_pstdev(times, trim_ratio=trim_ratio)*1000:.3f}")

    if args.query in ("count", "sum"):
        # Error stats for count/sum
        rel_trim = _trim_finite(rel_errors, trim_ratio=trim_ratio)
        print(f"avg_rel_error : {_finite_mean(rel_errors, trim_ratio=trim_ratio):.6f}")
        if len(rel_trim) > 1:
            print(f"std_rel_error : {_finite_pstdev(rel_errors, trim_ratio=trim_ratio):.6f}")
    else:
        # Rank-based stats for max
        nm_trim = _trim_finite(noisy_maxes, trim_ratio=trim_ratio)
        re_trim = _trim_finite(rank_errors, trim_ratio=trim_ratio)
        rr_trim = _trim_finite(rank_ratios, trim_ratio=trim_ratio)

        print(f"avg_noisy_max : {_finite_mean(noisy_maxes, trim_ratio=trim_ratio):.6f}")
        if len(nm_trim) > 1:
            print(f"std_noisy_max : {_finite_pstdev(noisy_maxes, trim_ratio=trim_ratio):.6f}")

        print(f"avg_rank_error: {_finite_mean(rank_errors, trim_ratio=trim_ratio):.3f}")
        if len(re_trim) > 1:
            print(f"std_rank_error: {_finite_pstdev(rank_errors, trim_ratio=trim_ratio):.3f}")

        print(f"avg_rank_ratio: {_finite_mean(rank_ratios, trim_ratio=trim_ratio):.6f}")
        if len(rr_trim) > 1:
            print(f"std_rank_ratio: {_finite_pstdev(rank_ratios, trim_ratio=trim_ratio):.6f}")

    # Append one summary row to <dataset_dir>/{count|sum|max}.csv
    avg_time_ms = _finite_mean(times, trim_ratio=trim_ratio) * 1000
    avg_result = _finite_mean(noisy_results, trim_ratio=trim_ratio)
    avg_abs_error = _finite_mean(abs_errors, trim_ratio=trim_ratio)
    avg_rel_error = _finite_mean(rel_errors, trim_ratio=trim_ratio)
    out_csv = Path(os.path.abspath(args.dataset_file)).parent / f"{args.query}.csv"
    _append_csv_row(
        out_csv,
        header=["ground_truth", "avg_result", "abs_error", "rel_error", "running time"],
        row=[truth, avg_result, avg_abs_error, avg_rel_error, avg_time_ms],
    )
    print(f"\n[CSV] Appended summary row to: {out_csv}")


if __name__ == "__main__":
    main()
