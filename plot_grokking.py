#!/usr/bin/env python3
"""Plot grokking training logs and summarize accuracy milestones."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_COLUMNS = (
    "step",
    "weight_norm",
    "train_loss",
    "train_exact_acc",
    "test_loss",
    "test_gen_in_set_token_acc",
    "test_set_acc",
    "test_exact_acc",
)


class LogError(ValueError):
    """A user-facing error in a training log."""


@dataclass
class Run:
    path: Path
    label: str
    step: list[float]
    weight_norm: list[float]
    train_loss: list[float]
    train_exact_acc: list[float]
    test_loss: list[float]
    test_gen_in_set_token_acc: list[float]
    test_set_acc: list[float]
    test_exact_acc: list[float]


@dataclass(frozen=True)
class Score:
    train99: float | None
    test90: float | None
    test99: float | None
    gap_x: float | None
    final_train: float
    final_test: float
    final_norm: float


def _number(value: str | None, path: Path, row: int, column: str) -> float:
    if value is None or not value.strip():
        raise LogError(f"{path}: row {row}, column {column!r} is empty")
    try:
        number = float(value)
    except ValueError as exc:
        raise LogError(
            f"{path}: row {row}, column {column!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise LogError(
            f"{path}: row {row}, column {column!r} must be finite: {value!r}"
        )
    return number


def load_run(path: str | os.PathLike[str]) -> Run:
    log_path = Path(path)
    try:
        handle = log_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise LogError(f"cannot read {log_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LogError(f"{log_path}: empty file (CSV header is missing)")
        fieldnames = [name.strip() if name is not None else "" for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            raise LogError(f"{log_path}: CSV header contains duplicate columns")
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise LogError(f"{log_path}: missing required columns: {', '.join(missing)}")
        reader.fieldnames = fieldnames

        values = {name: [] for name in REQUIRED_COLUMNS}
        run_kinds: list[tuple[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise LogError(f"{log_path}: row {row_number} has more fields than the header")
            if not any(value and value.strip() for value in row.values()):
                continue
            for column in REQUIRED_COLUMNS:
                values[column].append(_number(row.get(column), log_path, row_number, column))
            task = (row.get("task") or "").strip()
            constraint = (row.get("output_constraint") or "").strip()
            kind = (task, constraint)
            if kind not in run_kinds:
                run_kinds.append(kind)

    if not values["step"]:
        raise LogError(f"{log_path}: CSV contains no data rows")
    order = sorted(range(len(values["step"])), key=values["step"].__getitem__)
    ordered_steps = [values["step"][index] for index in order]
    if len(set(ordered_steps)) != len(ordered_steps):
        raise LogError(f"{log_path}: contains duplicate steps")
    values = {
        name: [column[index] for index in order]
        for name, column in values.items()
    }
    if len(run_kinds) > 1:
        raise LogError(f"{log_path}: contains multiple task/constraint configurations")
    task, constraint = run_kinds[0]
    kind_label = " / ".join(value for value in (task, constraint) if value)
    label = f"{log_path.stem} / {kind_label}" if kind_label else log_path.stem
    return Run(path=log_path, label=label, **values)


def _first_step(steps: Sequence[float], values: Sequence[float], threshold: float) -> float | None:
    matches = [step for step, value in zip(steps, values) if value >= threshold]
    return min(matches) if matches else None


def score_run(run: Run) -> Score:
    train99 = _first_step(run.step, run.train_exact_acc, 0.99)
    test90 = _first_step(run.step, run.test_exact_acc, 0.90)
    test99 = _first_step(run.step, run.test_exact_acc, 0.99)
    gap_x = (
        test90 / train99
        if train99 is not None and test90 is not None and train99 > 0
        else None
    )
    final_index = max(range(len(run.step)), key=run.step.__getitem__)
    return Score(
        train99=train99,
        test90=test90,
        test99=test99,
        gap_x=gap_x,
        final_train=run.train_exact_acc[final_index],
        final_test=run.test_exact_acc[final_index],
        final_norm=run.weight_norm[final_index],
    )


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        direct = Path(pattern)
        if direct.is_file():
            matches = [str(direct)]
        else:
            matches = sorted(glob.glob(pattern, recursive=True))
        files = [
            Path(match) for match in matches
            if Path(match).is_file() and Path(match).name != "sweep_summary.csv"
        ]
        if not files:
            raise LogError(f"input path/glob matched no files: {pattern}")
        for path in files:
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:g}"


def print_summary(runs: Sequence[Run]) -> None:
    headers = ("run", "train>=.99", "test>=.90", "test>=.99", "gap x", "final train", "final test", "final norm")
    rows = []
    for run in runs:
        score = score_run(run)
        rows.append(
            (
                run.label,
                _format_number(score.train99),
                _format_number(score.test90),
                _format_number(score.test99),
                _format_number(score.gap_x),
                _format_number(score.final_train),
                _format_number(score.final_test),
                _format_number(score.final_norm),
            )
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def plot_runs(runs: Sequence[Run], out: Path, title: str | None, linear_x: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise LogError("plotting requires matplotlib (install it with 'pip install matplotlib')") from exc

    if not linear_x:
        invalid = [run.label for run in runs if not any(step > 0 for step in run.step)]
        if invalid:
            raise LogError(
                "log x-axis requires a positive step in every run; use --linear-x "
                f"(no positive step: {', '.join(invalid)})"
            )

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharex=True)
    colors = plt.get_cmap("tab10")
    positive_losses = True
    for index, run in enumerate(runs):
        color = colors(index % 10)
        points = [i for i, step in enumerate(run.step) if linear_x or step > 0]
        x = [run.step[i] for i in points]
        train_exact = [run.train_exact_acc[i] for i in points]
        test_exact = [run.test_exact_acc[i] for i in points]
        test_in_set = [run.test_gen_in_set_token_acc[i] for i in points]
        test_set = [run.test_set_acc[i] for i in points]
        train_loss = [run.train_loss[i] for i in points]
        test_loss = [run.test_loss[i] for i in points]
        norm = [run.weight_norm[i] for i in points]
        positive_losses = positive_losses and all(value > 0 for value in train_loss + test_loss)

        axes[0].plot(x, train_exact, "--", color=color, label=f"{run.label} train")
        axes[0].plot(x, test_exact, "-", color=color, label=f"{run.label} test")
        axes[1].plot(x, test_in_set, ":", color=color, label=f"{run.label} in-set token")
        axes[1].plot(x, test_set, "--", color=color, label=f"{run.label} set")
        axes[1].plot(x, test_exact, "-", color=color, label=f"{run.label} exact")
        axes[2].plot(x, train_loss, "--", color=color, label=f"{run.label} train")
        axes[2].plot(x, test_loss, "-", color=color, label=f"{run.label} test")
        axes[3].plot(x, norm, color=color, label=run.label)

    axes[0].set_title("Exact accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Test generation decomposition")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_title("Loss")
    axes[2].set_ylabel("Loss")
    axes[3].set_title("Parameter L2 norm")
    axes[3].set_ylabel("L2 norm")
    if positive_losses:
        axes[2].set_yscale("log")
    else:
        print("warning: nonpositive loss found; using a linear loss axis", file=sys.stderr)

    for axis in axes:
        if not linear_x:
            axis.set_xscale("log")
        axis.set_xlabel("Step")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize="small")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    try:
        fig.savefig(out, format="png", dpi=160)
    except OSError as exc:
        raise LogError(f"cannot write {out}: {exc}") from exc
    finally:
        plt.close(fig)


def selftest() -> None:
    columns = [
        "step", "lr", "weight_norm", "train_loss", "train_token_acc",
        "train_gen_in_set_token_acc", "train_set_acc", "train_exact_acc",
        "test_loss", "test_token_acc", "test_gen_in_set_token_acc",
        "test_set_acc", "test_exact_acc", "elapsed_seconds", "task",
        "output_constraint",
    ]
    rows = [
        [0, .001, 2.0, 1.0, .1, .4, .1, .2, 1.2, .1, .3, .1, .1, 0, "sort", "free"],
        [20, .001, 2.5, .01, 1, 1, 1, 1, .4, .8, .95, .92, .91, 2, "sort", "free"],
        [10, .001, 2.3, .1, 1, 1, 1, .995, .7, .5, .8, .6, .4, 1, "sort", "free"],
        [30, .001, 2.7, .001, 1, 1, 1, 1, .01, 1, 1, 1, .995, 3, "sort", "free"],
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        run = load_run(path)
        score = score_run(run)
        assert run.label == "sample / sort / free"
        assert score.train99 == 10
        assert score.test90 == 20
        assert score.test99 == 30
        assert score.gap_x == 2
        assert score.final_train == 1
        assert score.final_test == .995
        assert score.final_norm == 2.7
    print("selftest passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("inputs", nargs="*", help="CSV paths or glob patterns")
    parser.add_argument("--out", default="grokking.png", help="output PNG (default: grokking.png)")
    parser.add_argument("--title", help="optional figure title")
    parser.add_argument("--linear-x", action="store_true", help="use a linear x-axis instead of log")
    parser.add_argument("--selftest", action="store_true", help="test CSV parsing and milestone scoring")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        try:
            selftest()
        except (AssertionError, LogError) as exc:
            print(f"selftest failed: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.inputs:
        parser.error("at least one CSV path or glob is required (or use --selftest)")

    try:
        runs = [load_run(path) for path in expand_inputs(args.inputs)]
        print_summary(runs)
        plot_runs(runs, Path(args.out), args.title, args.linear_x)
    except LogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
