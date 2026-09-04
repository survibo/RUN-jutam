#!/usr/bin/env python3
"""Plot legacy and EOS-model grokking logs and summarize milestones."""

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


CORE_COLUMNS = (
    "step",
    "weight_norm",
    "train_loss",
    "train_exact_acc",
    "test_loss",
    "test_exact_acc",
)
TOKEN_COLUMNS = ("train_token_acc", "test_token_acc")
VALIDATION_COLUMNS = (
    "validation_loss",
    "validation_token_acc",
    "validation_exact_acc",
)
GENERATION_COLUMNS = ("test_gen_in_set_token_acc", "test_set_acc")
STRATA = ("direct", "transitive", "unresolved")
STRATA_COLUMNS = tuple(
    column
    for name in STRATA
    for column in (f"test_{name}_exact_acc", f"test_{name}_count")
)


class LogError(ValueError):
    """A user-facing error in a training log."""


@dataclass
class Run:
    path: Path
    label: str
    schema: str
    step: list[float]
    weight_norm: list[float]
    train_loss: list[float]
    train_token_acc: list[float]
    train_exact_acc: list[float]
    validation_loss: list[float]
    validation_token_acc: list[float]
    validation_exact_acc: list[float]
    test_loss: list[float]
    test_token_acc: list[float]
    test_gen_in_set_token_acc: list[float]
    test_set_acc: list[float]
    test_exact_acc: list[float]
    test_direct_exact_acc: list[float]
    test_direct_count: list[float]
    test_transitive_exact_acc: list[float]
    test_transitive_count: list[float]
    test_unresolved_exact_acc: list[float]
    test_unresolved_count: list[float]


@dataclass(frozen=True)
class Score:
    train99: float | None
    validation90: float | None
    validation99: float | None
    test90: float | None
    test99: float | None
    gap_x: float | None
    final_train: float
    final_validation: float | None
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


def _optional_number(value: str | None, path: Path, row: int, column: str) -> float:
    if value is None or not value.strip():
        return math.nan
    return _number(value, path, row, column)


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
        has_validation = any(name in fieldnames for name in VALIDATION_COLUMNS)
        has_generation = any(name in fieldnames for name in GENERATION_COLUMNS)
        has_strata = any(name in fieldnames for name in STRATA_COLUMNS)
        if has_validation:
            schema = "eos"
            required = CORE_COLUMNS + TOKEN_COLUMNS + VALIDATION_COLUMNS
        elif has_generation or has_strata:
            schema = "legacy"
            required = CORE_COLUMNS + TOKEN_COLUMNS + GENERATION_COLUMNS + STRATA_COLUMNS
        else:
            raise LogError(
                f"{log_path}: unrecognized CSV schema; expected validation columns "
                "or legacy generation/strata columns"
            )
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise LogError(f"{log_path}: missing required columns: {', '.join(missing)}")
        reader.fieldnames = fieldnames

        all_columns = (
            CORE_COLUMNS
            + TOKEN_COLUMNS
            + VALIDATION_COLUMNS
            + GENERATION_COLUMNS
            + STRATA_COLUMNS
        )
        values = {name: [] for name in all_columns}
        run_kinds: list[tuple[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise LogError(f"{log_path}: row {row_number} has more fields than the header")
            if not any(value and value.strip() for value in row.values()):
                continue
            for column in required:
                parser = _optional_number if column in STRATA_COLUMNS else _number
                values[column].append(parser(row.get(column), log_path, row_number, column))
            for column in all_columns:
                if column not in required:
                    values[column].append(_optional_number(row.get(column), log_path, row_number, column))
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
    return Run(path=log_path, label=label, schema=schema, **values)


def _first_step(steps: Sequence[float], values: Sequence[float], threshold: float) -> float | None:
    matches = [step for step, value in zip(steps, values) if value >= threshold]
    return min(matches) if matches else None


def score_run(run: Run) -> Score:
    train99 = _first_step(run.step, run.train_exact_acc, 0.99)
    validation90 = (
        _first_step(run.step, run.validation_exact_acc, 0.90)
        if run.schema == "eos"
        else None
    )
    validation99 = (
        _first_step(run.step, run.validation_exact_acc, 0.99)
        if run.schema == "eos"
        else None
    )
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
        validation90=validation90,
        validation99=validation99,
        test90=test90,
        test99=test99,
        gap_x=gap_x,
        final_train=run.train_exact_acc[final_index],
        final_validation=(
            run.validation_exact_acc[final_index] if run.schema == "eos" else None
        ),
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
    include_validation = any(run.schema == "eos" for run in runs)
    headers = ["run", "train>=.99"]
    if include_validation:
        headers.extend(("val>=.90", "val>=.99"))
    headers.extend(("test>=.90", "test>=.99", "gap x", "final train"))
    if include_validation:
        headers.append("final val")
    headers.extend(("final test", "final norm"))
    rows = []
    for run in runs:
        score = score_run(run)
        row = [run.label, _format_number(score.train99)]
        if include_validation:
            row.extend(
                (
                    _format_number(score.validation90),
                    _format_number(score.validation99),
                )
            )
        row.extend(
            (
                _format_number(score.test90),
                _format_number(score.test99),
                _format_number(score.gap_x),
                _format_number(score.final_train),
            )
        )
        if include_validation:
            row.append(_format_number(score.final_validation))
        row.extend(
            (_format_number(score.final_test), _format_number(score.final_norm))
        )
        rows.append(row)
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

    has_eos = any(run.schema == "eos" for run in runs)
    has_legacy = any(run.schema == "legacy" for run in runs)
    panel_names = ["exact"]
    if has_eos:
        panel_names.append("token")
    if has_legacy:
        panel_names.extend(("generation", "strata"))
    panel_names.extend(("loss", "norm"))
    fig, axes = plt.subplots(
        1, len(panel_names), figsize=(4.6 * len(panel_names), 4.8), sharex=True
    )
    axis = dict(zip(panel_names, axes))
    colors = plt.get_cmap("tab10")
    positive_losses = True
    for index, run in enumerate(runs):
        color = colors(index % 10)
        points = [i for i, step in enumerate(run.step) if linear_x or step > 0]
        x = [run.step[i] for i in points]
        train_exact = [run.train_exact_acc[i] for i in points]
        test_exact = [run.test_exact_acc[i] for i in points]
        train_loss = [run.train_loss[i] for i in points]
        test_loss = [run.test_loss[i] for i in points]
        norm = [run.weight_norm[i] for i in points]
        losses = train_loss + test_loss
        axis["loss"].plot(
            x, train_loss, "--", color=color, label=f"{run.label} train"
        )

        axis["exact"].plot(
            x, train_exact, "--", color=color, label=f"{run.label} train"
        )
        if run.schema == "eos":
            validation_exact = [run.validation_exact_acc[i] for i in points]
            axis["exact"].plot(
                x,
                validation_exact,
                "-.",
                color=color,
                label=f"{run.label} validation",
            )
        axis["exact"].plot(
            x, test_exact, "-", color=color, label=f"{run.label} test"
        )

        if run.schema == "eos":
            axis["token"].plot(
                x,
                [run.train_token_acc[i] for i in points],
                "--",
                color=color,
                label=f"{run.label} train",
            )
            axis["token"].plot(
                x,
                [run.validation_token_acc[i] for i in points],
                "-.",
                color=color,
                label=f"{run.label} validation",
            )
            axis["token"].plot(
                x,
                [run.test_token_acc[i] for i in points],
                "-",
                color=color,
                label=f"{run.label} test",
            )
            validation_loss = [run.validation_loss[i] for i in points]
            losses.extend(validation_loss)
            axis["loss"].plot(
                x,
                validation_loss,
                "-.",
                color=color,
                label=f"{run.label} validation",
            )

        if run.schema == "legacy":
            axis["generation"].plot(
                x,
                [run.test_gen_in_set_token_acc[i] for i in points],
                ":",
                color=color,
                label=f"{run.label} in-set token",
            )
            axis["generation"].plot(
                x,
                [run.test_set_acc[i] for i in points],
                "--",
                color=color,
                label=f"{run.label} set",
            )
            axis["generation"].plot(
                x, test_exact, "-", color=color, label=f"{run.label} exact"
            )
            styles = {"direct": ":", "transitive": "-", "unresolved": "--"}
            for name in STRATA:
                accuracy = getattr(run, f"test_{name}_exact_acc")
                counts = getattr(run, f"test_{name}_count")
                stratum_points = [
                    i
                    for i in points
                    if math.isfinite(accuracy[i]) and counts[i] > 0
                ]
                if stratum_points:
                    axis["strata"].plot(
                        [run.step[i] for i in stratum_points],
                        [accuracy[i] for i in stratum_points],
                        styles[name],
                        color=color,
                        label=f"{run.label} {name}",
                    )

        positive_losses = positive_losses and all(value > 0 for value in losses)
        axis["loss"].plot(
            x, test_loss, "-", color=color, label=f"{run.label} test"
        )
        axis["norm"].plot(x, norm, color=color, label=run.label)

    axis["exact"].set_title("Exact accuracy")
    axis["exact"].set_ylabel("Accuracy")
    axis["exact"].set_ylim(-0.02, 1.02)
    if has_eos:
        axis["token"].set_title("Teacher-forced token accuracy")
        axis["token"].set_ylabel("Accuracy")
        axis["token"].set_ylim(-0.02, 1.02)
    if has_legacy:
        axis["generation"].set_title("Test generation decomposition")
        axis["generation"].set_ylabel("Accuracy")
        axis["generation"].set_ylim(-0.02, 1.02)
        axis["strata"].set_title("Test strata exact")
        axis["strata"].set_ylabel("Accuracy")
        axis["strata"].set_ylim(-0.02, 1.02)
    axis["loss"].set_title("Loss")
    axis["loss"].set_ylabel("Loss")
    axis["norm"].set_title("Parameter L2 norm")
    axis["norm"].set_ylabel("L2 norm")
    if positive_losses:
        axis["loss"].set_yscale("log")
    else:
        print("warning: nonpositive loss found; using a linear loss axis", file=sys.stderr)

    for current_axis in axes:
        if not linear_x:
            current_axis.set_xscale("log")
        current_axis.set_xlabel("Step")
        current_axis.grid(True, which="both", alpha=0.25)
        if current_axis.lines:
            current_axis.legend(fontsize="small")
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
        "output_constraint", "test_direct_exact_acc", "test_direct_count",
        "test_transitive_exact_acc", "test_transitive_count",
        "test_unresolved_exact_acc", "test_unresolved_count",
    ]
    rows = [
        [0, .001, 2.0, 1.0, .1, .4, .1, .2, 1.2, .1, .3, .1, .1, 0, "sort", "free", .2, 4, .1, 4, "", 0],
        [20, .001, 2.5, .01, 1, 1, 1, 1, .4, .8, .95, .92, .91, 2, "sort", "free", .95, 4, .9, 4, "", 0],
        [10, .001, 2.3, .1, 1, 1, 1, .995, .7, .5, .8, .6, .4, 1, "sort", "free", .5, 4, .3, 4, "", 0],
        [30, .001, 2.7, .001, 1, 1, 1, 1, .01, 1, 1, 1, .995, 3, "sort", "free", 1, 4, .99, 4, "", 0],
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        run = load_run(path)
        score = score_run(run)
        assert run.schema == "legacy"
        assert run.label == "sample / sort / free"
        assert score.train99 == 10
        assert score.test90 == 20
        assert score.test99 == 30
        assert score.gap_x == 2
        assert score.final_train == 1
        assert score.final_test == .995
        assert score.final_norm == 2.7
        eos_columns = [
            "step", "epoch", "lr", "weight_norm",
            "train_loss", "train_token_acc", "train_exact_acc",
            "validation_loss", "validation_token_acc", "validation_exact_acc",
            "test_loss", "test_token_acc", "test_exact_acc",
            "elapsed_seconds", "train_eval_count", "validation_eval_count",
            "test_eval_count", "run_signature_sha256",
        ]
        eos_rows = [
            [1, .1, .001, 2.0, 1.0, .5, .5, 1.1, .4, 0, 1.2, .3, 0, 0, 10, 10, 10, "a"],
            [20, 2, .001, 2.7, .01, 1, 1, .02, 1, 1, .03, .95, .92, 2, 10, 10, 10, "a"],
            [10, 1, .001, 2.4, .1, 1, 1, .2, .95, .91, .3, .8, .5, 1, 10, 10, 10, "a"],
        ]
        eos_path = Path(directory) / "eos.csv"
        with eos_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(eos_columns)
            writer.writerows(eos_rows)
        eos_run = load_run(eos_path)
        eos_score = score_run(eos_run)
        assert eos_run.schema == "eos"
        assert eos_run.step == [1, 10, 20]
        assert eos_score.train99 == 10
        assert eos_score.validation90 == 10
        assert eos_score.validation99 == 20
        assert eos_score.test90 == 20
        assert eos_score.gap_x == 2
        assert eos_score.final_validation == 1
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
