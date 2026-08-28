#!/usr/bin/env python3
"""Run a sequential hyperparameter sweep for sortformer.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TASKS = ("ascending", "mod", "alternating")
OUTPUT_CONSTRAINTS = ("permutation", "input-only", "free")
SUMMARY_FIELDS = (
    "task",
    "train_percent",
    "dataset",
    "weight_decay",
    "output_constraint",
    "runs",
    "successful_test90",
    "median_grokking_gap",
    "min_grokking_gap",
    "max_grokking_gap",
    "median_final_train_exact",
    "median_final_test_exact",
    "csv_paths",
    "checkpoint_paths",
)
MANAGED_OPTIONS = {
    "--data", "--task", "--n", "--m", "--train-percent", "--modulus",
    "--data-seed", "--n-test", "--seed", "--weight-decay",
    "--output-constraint", "--steps", "--log-csv", "--out-dir", "--resume",
    "--smoke",
}


class SweepError(ValueError):
    """A user-facing sweep configuration or artifact error."""


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    config: dict[str, Any]
    command: list[str]
    csv_path: Path
    checkpoint_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def unique(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def number_label(value: float) -> str:
    return format(value, ".12g").replace("-", "neg").replace(".", "p")


def slug(value: str, limit: int = 32) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part)[:limit] or "data"


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded = list(args.sortformer_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    for token in forwarded:
        option = token.split("=", 1)[0]
        if option in MANAGED_OPTIONS:
            raise SweepError(
                f"{option} is managed by sweep.py and cannot be passed after --"
            )
    return forwarded


def build_runs(args: argparse.Namespace, out_dir: Path) -> list[RunSpec]:
    forwarded = forwarded_args(args)
    tasks = unique(args.tasks)
    seeds = unique(args.seeds)
    weight_decays = unique(args.weight_decays)
    constraints = unique(args.output_constraints)
    if args.data is not None:
        if len(args.train_percents) != 1:
            raise SweepError("with --data, --train-percents must contain exactly one value")
        data_path = args.data.expanduser().resolve()
        if not data_path.is_dir():
            raise SweepError(f"dataset directory does not exist: {data_path}")
        train_percents: list[float | None] = [None]
        dataset = str(data_path)
        dataset_label = f"data-{slug(data_path.name)}"
    else:
        data_path = None
        train_percents = unique(args.train_percents)
        dataset = "generated"
        dataset_label = "generated"

    script = Path(__file__).resolve().with_name("sortformer.py")
    specs: list[RunSpec] = []
    used_ids: dict[str, dict[str, Any]] = {}
    combinations = itertools.product(
        tasks, train_percents, weight_decays, constraints, seeds
    )
    for task, train_percent, weight_decay, constraint, seed in combinations:
        config: dict[str, Any] = {
            "task": task,
            "seed": seed,
            "weight_decay": weight_decay,
            "output_constraint": constraint,
            "steps": args.steps,
            "dataset": dataset,
            "train_percent": train_percent,
            "forwarded_args": forwarded,
        }
        if data_path is None:
            config.update({
                "n": args.n,
                "m": args.m,
                "modulus": args.modulus,
                "data_seed": args.data_seed,
                "n_test": args.n_test,
            })
        digest = config_digest(config)
        percent_label = (
            f"tp-{number_label(train_percent)}"
            if train_percent is not None else dataset_label
        )
        prefix = (
            f"{task}__{percent_label}__wd-{number_label(weight_decay)}"
            f"__oc-{constraint}__seed-{seed}"
        )
        digest_length = 12
        run_id = f"{prefix}__{digest[:digest_length]}"
        while run_id in used_ids and used_ids[run_id] != config:
            digest_length += 4
            run_id = f"{prefix}__{digest[:digest_length]}"
        used_ids[run_id] = config

        csv_path = out_dir / f"{run_id}.csv"
        checkpoint_path = out_dir / run_id
        command = [
            args.python,
            str(script),
            "--task", task,
            "--seed", str(seed),
            "--weight-decay", format(weight_decay, ".17g"),
            "--output-constraint", constraint,
            "--steps", str(args.steps),
            "--log-csv", str(csv_path),
            "--out-dir", str(checkpoint_path),
        ]
        if data_path is not None:
            command.extend(("--data", str(data_path)))
        else:
            command.extend((
                "--n", str(args.n), "--m", str(args.m),
                "--train-percent", format(train_percent, ".17g"),
                "--modulus", str(args.modulus), "--data-seed", str(args.data_seed),
                "--n-test", str(args.n_test),
            ))
        command.extend(forwarded)
        specs.append(RunSpec(run_id, config, command, csv_path, checkpoint_path))
    return specs


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "runs": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise SweepError(f"invalid manifest structure: {path}")
    return value


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=".manifest-",
            suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def manifest_entry(spec: RunSpec, status: str = "planned") -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "command": spec.command,
        "config": spec.config,
        "status": status,
        "returncode": None,
        "csv_path": str(spec.csv_path),
        "checkpoint_path": str(spec.checkpoint_path),
    }


def artifact_state(spec: RunSpec) -> tuple[bool, bool]:
    return spec.csv_path.is_file(), (spec.checkpoint_path / "ckpt_final.pt").is_file()


def completed_manifest_specs(entries: dict[str, dict[str, Any]]) -> list[RunSpec]:
    specs = []
    for run_id, entry in entries.items():
        if entry.get("status") not in {"completed", "existing"}:
            continue
        try:
            spec = RunSpec(
                run_id=run_id,
                config=entry["config"],
                command=list(entry["command"]),
                csv_path=Path(entry["csv_path"]),
                checkpoint_path=Path(entry["checkpoint_path"]),
            )
        except (KeyError, TypeError):
            continue
        has_csv, has_checkpoint = artifact_state(spec)
        if has_csv and has_checkpoint:
            specs.append(spec)
    return specs


def finite_float(value: str | None, path: Path, row: int, field: str) -> float:
    try:
        result = float(value if value is not None else "")
    except ValueError as exc:
        raise SweepError(f"{path}: row {row} has invalid {field}") from exc
    if not math.isfinite(result):
        raise SweepError(f"{path}: row {row} has non-finite {field}")
    return result


def score_csv(path: Path) -> dict[str, float | None]:
    required = {"step", "train_exact_acc", "test_exact_acc"}
    points: list[tuple[float, float, float]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise SweepError(f"cannot read run CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise SweepError(f"{path}: missing CSV columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            points.append((
                finite_float(row.get("step"), path, row_number, "step"),
                finite_float(row.get("train_exact_acc"), path, row_number, "train_exact_acc"),
                finite_float(row.get("test_exact_acc"), path, row_number, "test_exact_acc"),
            ))
    if not points:
        raise SweepError(f"{path}: CSV contains no data rows")
    train99_steps = [step for step, train, _ in points if train >= 0.99]
    test90_steps = [step for step, _, test in points if test >= 0.90]
    train99 = min(train99_steps) if train99_steps else None
    test90 = min(test90_steps) if test90_steps else None
    final = max(points, key=lambda point: point[0])
    gap = (
        test90 / train99
        if train99 is not None and test90 is not None and train99 > 0 else None
    )
    return {
        "test90": test90,
        "gap": gap,
        "final_train": final[1],
        "final_test": final[2],
    }


def metric(value: float | None) -> str:
    return "" if value is None else format(value, ".10g")


def aggregate(specs: Sequence[RunSpec], entries: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[Any, ...], list[tuple[RunSpec, dict[str, float | None]]]] = {}
    for spec in specs:
        entry = entries[spec.run_id]
        if entry["status"] not in {"completed", "existing"}:
            continue
        score = score_csv(spec.csv_path)
        config = spec.config
        key = (
            config["task"], config["train_percent"], config["dataset"],
            config["weight_decay"], config["output_constraint"],
        )
        groups.setdefault(key, []).append((spec, score))

    rows: list[dict[str, str]] = []
    for key in sorted(groups, key=lambda item: tuple("" if x is None else str(x) for x in item)):
        task, train_percent, dataset, weight_decay, constraint = key
        runs = groups[key]
        gaps = [score["gap"] for _, score in runs if score["gap"] is not None]
        final_train = [score["final_train"] for _, score in runs]
        final_test = [score["final_test"] for _, score in runs]
        rows.append({
            "task": str(task),
            "train_percent": metric(train_percent),
            "dataset": str(dataset) if train_percent is None else "",
            "weight_decay": metric(weight_decay),
            "output_constraint": str(constraint),
            "runs": str(len(runs)),
            "successful_test90": str(sum(score["test90"] is not None for _, score in runs)),
            "median_grokking_gap": metric(statistics.median(gaps) if gaps else None),
            "min_grokking_gap": metric(min(gaps) if gaps else None),
            "max_grokking_gap": metric(max(gaps) if gaps else None),
            "median_final_train_exact": metric(statistics.median(final_train)),
            "median_final_test_exact": metric(statistics.median(final_test)),
            "csv_paths": ";".join(str(spec.csv_path) for spec, _ in runs),
            "checkpoint_paths": ";".join(str(spec.checkpoint_path) for spec, _ in runs),
        })
    return rows


def write_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        print("summary: no completed runs")
        return
    print("summary: task dataset/tp wd constraint runs test90 gap(median[min,max]) final(train,test)")
    for row in rows:
        dataset = row["train_percent"] or row["dataset"]
        gap = row["median_grokking_gap"] or "-"
        bounds = (
            f"[{row['min_grokking_gap']},{row['max_grokking_gap']}]"
            if row["min_grokking_gap"] else ""
        )
        print(
            f"  {row['task']} {dataset} {row['weight_decay']} {row['output_constraint']} "
            f"{row['runs']} {row['successful_test90']} {gap}{bounds} "
            f"({row['median_final_train_exact']},{row['median_final_test_exact']})"
        )


def selftest() -> None:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = parser.parse_args([
            "--tasks", "ascending", "mod", "--seeds", "1", "2",
            "--weight-decays", "0.1", "1", "--train-percents", "1", "2",
            "--output-constraints", "permutation", "free", "--out-dir", str(root),
            "--", "--batch-size", "64", "--device", "cpu",
        ])
        specs = build_runs(args, root)
        assert len(specs) == 32
        assert len({spec.run_id for spec in specs}) == 32
        assert all(spec.command[-4:] == ["--batch-size", "64", "--device", "cpu"] for spec in specs)
        assert all("--log-csv" in spec.command and "--out-dir" in spec.command for spec in specs)

        group_specs = [spec for spec in specs if (
            spec.config["task"] == "ascending"
            and spec.config["train_percent"] == 1.0
            and spec.config["weight_decay"] == 0.1
            and spec.config["output_constraint"] == "permutation"
        )]
        third = RunSpec(
            group_specs[0].run_id + "-third",
            {**group_specs[0].config, "seed": 3},
            group_specs[0].command,
            root / "third.csv",
            root / "third",
        )
        test_specs = group_specs + [third]
        csv_rows = (
            [(10, .995, .2), (40, 1, .91)],
            [(20, .99, .90)],
            [(10, 1, .2), (40, 1, .70)],
        )
        entries: dict[str, dict[str, Any]] = {}
        for spec, points in zip(test_specs, csv_rows):
            with spec.csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("step", "train_exact_acc", "test_exact_acc"))
                writer.writerows(points)
            entries[spec.run_id] = manifest_entry(spec, "completed")
        rows = aggregate(test_specs, entries)
        assert len(rows) == 1
        assert rows[0]["runs"] == "3" and rows[0]["successful_test90"] == "2"
        assert rows[0]["median_grokking_gap"] == "2.5"
        assert rows[0]["min_grokking_gap"] == "1" and rows[0]["max_grokking_gap"] == "4"
        assert rows[0]["median_final_test_exact"] == "0.9"
    print("sweep selftest passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--modulus", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["ascending"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1.0])
    parser.add_argument("--train-percents", nargs="+", type=float, default=[2.0])
    parser.add_argument(
        "--output-constraints", nargs="+", choices=OUTPUT_CONSTRAINTS,
        default=["permutation"],
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--out-dir", type=Path, default=Path("sweeps"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("sortformer_args", nargs=argparse.REMAINDER, help="arguments after -- for sortformer.py")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise SweepError("--steps must be positive")
    if any(not math.isfinite(value) or value < 0 for value in args.weight_decays):
        raise SweepError("--weight-decays must be finite and nonnegative")
    if args.data is None:
        if args.n < 1 or args.m < 1 or args.m > args.n:
            raise SweepError("require 1 <= --m <= --n")
        if args.modulus < 1:
            raise SweepError("--modulus must be positive")
        if args.n_test < -1:
            raise SweepError("--n-test must be -1 or nonnegative")
        if any(not math.isfinite(value) or not 0 < value < 100 for value in args.train_percents):
            raise SweepError("--train-percents must be finite and between 0 and 100")


def run_sweep(args: argparse.Namespace) -> int:
    validate_args(args)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_runs(args, out_dir)
    manifest_path = out_dir / "sweep_manifest.json"
    summary_path = out_dir / "sweep_summary.csv"
    if not args.dry_run:
        for spec in specs:
            has_csv, has_checkpoint = artifact_state(spec)
            if not (has_csv or has_checkpoint):
                continue
            if args.skip_existing and has_csv and has_checkpoint:
                continue
            kind = "complete" if has_csv and has_checkpoint else "partial"
            raise SweepError(
                f"{kind} artifacts already exist for {spec.run_id}; use --skip-existing "
                "for complete runs or remove/relocate artifacts"
            )

    manifest = load_manifest(manifest_path)
    existing_entries = {
        entry.get("run_id"): entry for entry in manifest["runs"]
        if isinstance(entry, dict) and isinstance(entry.get("run_id"), str)
    }
    entries = dict(existing_entries)
    for spec in specs:
        prior = entries.get(spec.run_id)
        if prior is not None and prior.get("config") != spec.config:
            raise SweepError(f"run ID collision with manifest entry: {spec.run_id}")
        if prior is None:
            entries[spec.run_id] = manifest_entry(spec)
    manifest["runs"] = list(entries.values())
    write_manifest(manifest_path, manifest)

    failed = False
    interrupted = False
    for index, spec in enumerate(specs, start=1):
        entry = entries[spec.run_id]
        if args.dry_run:
            if entry.get("status") not in {"completed", "existing"}:
                entry.update(status="dry-run", returncode=None)
            print(f"[{index}/{len(specs)}] dry-run {subprocess.list2cmdline(spec.command)}")
            write_manifest(manifest_path, manifest)
            continue
        has_csv, has_checkpoint = artifact_state(spec)
        if has_csv or has_checkpoint:
            if args.skip_existing and has_csv and has_checkpoint:
                entry.update(status="existing", returncode=0, finished_at=utc_now())
                print(f"[{index}/{len(specs)}] existing {spec.run_id}")
                write_manifest(manifest_path, manifest)
                continue
        entry.update(status="running", returncode=None, started_at=utc_now())
        write_manifest(manifest_path, manifest)
        print(f"[{index}/{len(specs)}] running {spec.run_id}", flush=True)
        try:
            result = subprocess.run(spec.command, shell=False, check=False)
            returncode = result.returncode
            has_csv, has_checkpoint = artifact_state(spec)
            completed = returncode == 0 and has_csv and has_checkpoint
            entry.update(
                status="completed" if completed else "failed",
                returncode=returncode,
                finished_at=utc_now(),
            )
            if returncode == 0 and not completed:
                entry["error"] = "training returned success but expected artifacts are missing"
            failed = failed or not completed
        except KeyboardInterrupt:
            entry.update(status="interrupted", returncode=None, finished_at=utc_now())
            interrupted = True
        except OSError as exc:
            entry.update(status="failed", returncode=None, error=str(exc), finished_at=utc_now())
            failed = True
        finally:
            write_manifest(manifest_path, manifest)
        if interrupted or (entry["status"] == "failed" and not args.continue_on_error):
            break

    aggregation_failed = False
    try:
        rows = aggregate(completed_manifest_specs(entries), entries)
    except SweepError as exc:
        aggregation_failed = True
        rows = []
        print(f"aggregation error: {exc}", file=sys.stderr)
    write_summary(summary_path, rows)
    print_summary(rows)
    print(f"manifest: {manifest_path}")
    print(f"summary CSV: {summary_path}")
    if interrupted:
        return 130
    return 1 if failed or aggregation_failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    try:
        return run_sweep(args)
    except (OSError, SweepError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
