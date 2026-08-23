#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import FD


RECORD_MAPPING_CANDIDATES = ["record.csv"]
USER_MAPPING_CANDIDATES = ["user.csv"]


def _trim_finite(xs: List[float], trim_ratio: float = 0.0) -> List[float]:
    xs2 = sorted(x for x in xs if isinstance(x, (int, float)) and math.isfinite(x))
    if not xs2 or trim_ratio <= 0:
        return xs2
    k = int(math.floor(len(xs2) * trim_ratio))
    return xs2[k:len(xs2) - k] if 2 * k < len(xs2) else xs2


def _finite_mean(xs: List[float], trim_ratio: float = 0.0) -> float:
    xs2 = _trim_finite(xs, trim_ratio)
    return statistics.mean(xs2) if xs2 else float("nan")


def _finite_pstdev(xs: List[float], trim_ratio: float = 0.0) -> float:
    xs2 = _trim_finite(xs, trim_ratio)
    return statistics.pstdev(xs2) if len(xs2) > 1 else float("nan")


def _append_csv_row(csv_path: Path, header: List[str], row: List[Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _parse_value(s: str) -> Union[float, str]:
    try:
        return float(s)
    except ValueError:
        return s


def _split_parts(line: str) -> List[str]:
    return line.replace(",", " ").split()


def _read_mapping_file(path: str, value_name: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _split_parts(line)
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected 'id {value_name}'")
            mapping[parts[0]] = float(parts[1])
    return mapping


def _find_sibling_file(dataset_path: str, candidates: List[str]) -> str:
    directory = os.path.dirname(os.path.abspath(dataset_path)) or "."
    for name in candidates:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find required sibling file in {directory}: {candidates}")


def preprocess(dataset_path: str) -> Tuple[FD.Structure, int, str, str]:
    record_mapping_path = _find_sibling_file(dataset_path, RECORD_MAPPING_CANDIDATES)
    user_mapping_path = _find_sibling_file(dataset_path, USER_MAPPING_CANDIDATES)
    record_budget_map = _read_mapping_file(record_mapping_path, "record_privacy_budget")
    user_budget_map = _read_mapping_file(user_mapping_path, "user_privacy_budget")

    user_records: Dict[str, List[str]] = defaultdict(list)
    record_values: Dict[str, Any] = {}
    n_total = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = _split_parts(line)
            if len(parts) < 3:
                raise ValueError(f"{dataset_path}:{lineno}: expected 'record_id user_id record_value'")
            rid, uid, raw_value = parts[:3]
            record_values[rid] = _parse_value(raw_value)
            user_records[uid].append(rid)
            n_total += 1

    users: FD.UsersDict = {
        uid: (float(user_budget_map.get(uid, 0.0)), record_ids)
        for uid, record_ids in user_records.items()
    }
    records: FD.RecordsDict = {
        rid: (value, float(record_budget_map.get(rid, 0.0)))
        for rid, value in record_values.items()
    }
    return (users, records), n_total, record_mapping_path, user_mapping_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file", help="Path to dataset_file (record_id user_id record_value per line)")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="Number of Folded-sum rounds")
    parser.add_argument("-beta", "--beta", type=float, default=0.1, help="Failure rate beta (0 < beta < 1)")
    args = parser.parse_args()

    structure, n_total, record_map_path, user_map_path = preprocess(args.dataset_file)
    if n_total <= 0:
        raise SystemExit("dataset_file contains 0 records (after removing empty/comment lines).")

    values = [float(value) for value, _budget in structure[1].values() if isinstance(value, (int, float))]
    if len(values) != len(structure[1]):
        raise ValueError("Folded-sum requires numeric record values.")
    truth = float(sum(values))

    print("=== Input ===")
    print(f"dataset_file        : {os.path.abspath(args.dataset_file)}")
    print(f"record_mapping_file : {record_map_path}")
    print(f"user_mapping_file   : {user_map_path}")
    print(f"total_records (N)   : {n_total}")
    print(f"rounds              : {args.rounds}")
    print(f"beta                : {args.beta}")
    print("tau_values          : 2^12 through 2^56 (45 values)")
    print(f"truth_sum           : {truth:.6f}\n")

    times: List[float] = []
    results: List[float] = []
    abs_errors: List[float] = []
    rel_errors: List[float] = []
    for round_number in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        noisy_sum = FD.pt(structure, beta=args.beta)
        elapsed = time.perf_counter() - t0
        abs_error = abs(noisy_sum - truth)
        rel_error = abs_error / abs(truth) if truth != 0 else float("nan")
        times.append(elapsed)
        results.append(noisy_sum)
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)
        print(
            f"[Round {round_number:>3}] noisy_sum={noisy_sum:.6f}  "
            f"time={elapsed * 1000:.3f} ms  rel_error={rel_error:.6f}"
        )

    trim_ratio = 0.2
    print("\n=== Summary ===")
    print(f"avg_time_ms   : {_finite_mean(times, trim_ratio) * 1000:.3f}")
    print(f"avg_rel_error : {_finite_mean(rel_errors, trim_ratio):.6f}")
    if len(_trim_finite(times, trim_ratio)) > 1:
        print(f"std_time_ms   : {_finite_pstdev(times, trim_ratio) * 1000:.3f}")
    if len(_trim_finite(rel_errors, trim_ratio)) > 1:
        print(f"std_rel_error : {_finite_pstdev(rel_errors, trim_ratio):.6f}")

    out_csv = Path(os.path.abspath(args.dataset_file)).parent / "sum.csv"
    _append_csv_row(
        out_csv,
        header=["ground_truth", "avg_result", "abs_error", "rel_error", "running time"],
        row=[
            truth,
            _finite_mean(results, trim_ratio),
            _finite_mean(abs_errors, trim_ratio),
            _finite_mean(rel_errors, trim_ratio),
            _finite_mean(times, trim_ratio) * 1000,
        ],
    )
    print(f"\n[CSV] Appended summary row to: {out_csv}")


if __name__ == "__main__":
    main()
