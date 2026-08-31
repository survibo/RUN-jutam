"""Deterministic random-order datasets for causal-GPT sorting experiments.

Each row contains one fixed-length sequence of distinct values in random order,
followed by the same three targets used by ``sortdata.py``.  Train and test are
still split by the underlying unordered combination, so different permutations
of the same values can never leak across splits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import sortdata
from sortdata import DatasetConfig


FORMAT_VERSION = 1
HEADER_PREFIX = "# gpt-sortdata-config: "
_TRAIN_ORDER_XOR = 0xA0761D6478BD642F
_TEST_ORDER_XOR = 0xE7037ED1A0B428DB


@dataclass(frozen=True)
class GPTSortExample:
    """One random-order input sequence and its three target sequences."""

    inputs: tuple[int, ...]
    asc: tuple[int, ...]
    mod: tuple[int, ...]
    alt: tuple[int, ...]


@dataclass(frozen=True)
class LoadedSplit:
    config: dict[str, object]
    split: str
    examples: tuple[GPTSortExample, ...]


@dataclass(frozen=True)
class LoadedDataset:
    metadata: dict[str, object]
    train: LoadedSplit
    test: LoadedSplit


def _permutation_seed(data_seed: int, rank: int, split: str) -> int:
    split_key = _TRAIN_ORDER_XOR if split == "train" else _TEST_ORDER_XOR
    # Per-rank mixing makes the permutation independent of row enumeration order.
    value = (data_seed ^ split_key ^ (rank * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def make_random_order_example(
    rank: int, config: DatasetConfig, split: str
) -> GPTSortExample:
    """Build one reproducibly shuffled input and its canonical targets."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    canonical = sortdata.make_example(
        sortdata.combination_unrank(rank, config.n, config.m), config.modulus
    )
    inputs = list(canonical.inputs)
    random.Random(_permutation_seed(config.seed, rank, split)).shuffle(inputs)
    return GPTSortExample(tuple(inputs), canonical.asc, canonical.mod, canonical.alt)


def iter_examples(
    ranks: Sequence[int], config: DatasetConfig, split: str
) -> Iterator[GPTSortExample]:
    for rank in ranks:
        yield make_random_order_example(rank, config, split)


def _format_example(example: GPTSortExample) -> str:
    fields = lambda values: " ".join(map(str, values))
    return (
        f"{fields(example.inputs)} -> asc: {fields(example.asc)} | "
        f"mod: {fields(example.mod)} | alt: {fields(example.alt)}"
    )


def _header(config: DatasetConfig, split: str, count: int, total: int) -> str:
    payload = asdict(config)
    payload.update(
        format_version=FORMAT_VERSION,
        dataset_type="causal-gpt-random-order",
        split=split,
        count=count,
        total=total,
        train_sizing_mode="count" if config.train_count is not None else "percent",
    )
    return HEADER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_split(
    path: Path, config: DatasetConfig, split: str, ranks: Sequence[int], total: int
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_header(config, split, len(ranks), total) + "\n")
        handle.write("# random input -> asc: ... | mod: ... | alt: ...\n")
        for example in iter_examples(ranks, config, split):
            handle.write(_format_example(example) + "\n")


def write_dataset(
    out: str | Path,
    config: DatasetConfig,
    train_ranks: Sequence[int],
    test_ranks: Sequence[int],
    total: int,
) -> dict[str, object]:
    """Write a causal-GPT dataset directory."""
    out_path = Path(out)
    if out_path.exists() and not out_path.is_dir():
        raise ValueError(f"output path is not a directory: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    _write_split(out_path / "train.txt", config, "train", train_ranks, total)
    _write_split(out_path / "test.txt", config, "test", test_ranks, total)
    coverage = sortdata.coverage_report(train_ranks, test_ranks, config)
    sizing_mode = "count" if config.train_count is not None else "percent"
    metadata = {
        "format_version": FORMAT_VERSION,
        "dataset_type": "causal-gpt-random-order",
        "config": asdict(config),
        "split_strategy": config.split_strategy,
        "train_sizing": {
            "mode": sizing_mode,
            "requested": (
                config.train_count
                if config.train_count is not None
                else config.train_percent
            ),
            "effective_count": len(train_ranks),
        },
        "total_combinations": total,
        "train_count": len(train_ranks),
        "test_count": len(test_ranks),
        "test_is_all_remaining": len(test_ranks) == total - len(train_ranks),
        "input_order": {
            "kind": "deterministic-random-permutation",
            "distinct_values": True,
            "fixed_length": config.m,
            "seed": config.seed,
        },
        "files": {"train": "train.txt", "test": "test.txt"},
        "coverage": coverage,
    }
    with (out_path / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return coverage


def _parse_values(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.strip().split())


def _parse_example(line: str) -> GPTSortExample:
    try:
        input_text, targets = line.split(" -> ", 1)
        labelled = dict(section.split(": ", 1) for section in targets.split(" | "))
        return GPTSortExample(
            _parse_values(input_text),
            _parse_values(labelled["asc"]),
            _parse_values(labelled["mod"]),
            _parse_values(labelled["alt"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid gpt-sortdata row: {line!r}") from exc


def _validate_example(example: GPTSortExample, config: DatasetConfig) -> None:
    if len(example.inputs) != config.m:
        raise ValueError(f"invalid input length: {example.inputs}")
    if len(set(example.inputs)) != config.m:
        raise ValueError(f"input values must be distinct: {example.inputs}")
    canonical_inputs = tuple(sorted(example.inputs))
    if any(value < 0 or value >= config.n for value in canonical_inputs):
        raise ValueError(f"input value outside [0, {config.n}): {example.inputs}")
    canonical = sortdata.make_example(canonical_inputs, config.modulus)
    if (example.asc, example.mod, example.alt) != (
        canonical.asc,
        canonical.mod,
        canonical.alt,
    ):
        raise ValueError(f"invalid targets for input: {example.inputs}")


def _validate_config(config: DatasetConfig) -> None:
    if config.n < 1:
        raise ValueError("n must be positive")
    if config.m < 1 or config.m > config.n:
        raise ValueError("m must satisfy 1 <= m <= n")
    if (config.train_percent is None) == (config.train_count is None):
        raise ValueError("exactly one of train-percent and train-count must be set")
    if config.train_percent is not None and (
        not math.isfinite(config.train_percent)
        or not 0 <= config.train_percent <= 100
    ):
        raise ValueError("train-percent must be finite and in [0, 100]")
    if config.train_count is not None and (
        type(config.train_count) is not int or config.train_count < 0
    ):
        raise ValueError("train-count must be a nonnegative integer")
    if config.split_strategy not in ("random", "relation-complete"):
        raise ValueError("split-strategy must be random or relation-complete")
    if config.modulus <= 0:
        raise ValueError("modulus must be positive")
    if config.n_test < -1:
        raise ValueError("n-test must be -1 or nonnegative")
    if config.enumerate_limit < 1:
        raise ValueError("enumerate-limit must be positive")


def load_split(path: str | Path, *, validate: bool = True) -> LoadedSplit:
    source = Path(path)
    config_payload: dict[str, object] | None = None
    examples: list[GPTSortExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if line.startswith(HEADER_PREFIX):
                config_payload = json.loads(line.removeprefix(HEADER_PREFIX))
            elif line and not line.startswith("#"):
                examples.append(_parse_example(line))
    if config_payload is None:
        raise ValueError(f"missing gpt-sortdata config header in {source}")
    if config_payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported gpt-sortdata format version in {source}")
    if config_payload.get("dataset_type") != "causal-gpt-random-order":
        raise ValueError(f"unexpected dataset type in {source}")
    config_keys = tuple(asdict(DatasetConfig()).keys())
    missing = [key for key in config_keys if key not in config_payload]
    if missing:
        raise ValueError(f"missing config fields in {source}: {', '.join(missing)}")
    config = DatasetConfig(**{key: config_payload[key] for key in config_keys})
    _validate_config(config)
    if int(config_payload.get("count", -1)) != len(examples):
        raise ValueError(f"row count does not match header in {source}")
    if validate:
        for example in examples:
            _validate_example(example, config)
    return LoadedSplit(config_payload, str(config_payload.get("split")), tuple(examples))


def load_dataset(path: str | Path, *, validate: bool = True) -> LoadedDataset:
    directory = Path(path)
    with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported gpt-sortdata metadata format version")
    if metadata.get("dataset_type") != "causal-gpt-random-order":
        raise ValueError("dataset was not generated by gpt_sortdata.py")
    files = metadata.get("files", {})
    train = load_split(directory / files.get("train", "train.txt"), validate=validate)
    test = load_split(directory / files.get("test", "test.txt"), validate=validate)
    if train.split != "train" or test.split != "test":
        raise ValueError("dataset split labels must be train and test")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("metadata config must be an object")
    config_keys = tuple(asdict(DatasetConfig()).keys())
    for split in (train, test):
        split_config = {key: split.config.get(key) for key in config_keys}
        if split_config != config:
            raise ValueError(f"{split.split} header does not match metadata config")
    if metadata.get("train_count") != len(train.examples) or metadata.get(
        "test_count"
    ) != len(test.examples):
        raise ValueError("metadata row counts do not match split files")
    train_sets = [tuple(sorted(example.inputs)) for example in train.examples]
    test_sets = [tuple(sorted(example.inputs)) for example in test.examples]
    if len(set(train_sets)) != len(train_sets):
        raise ValueError("duplicate input combination in train split")
    if len(set(test_sets)) != len(test_sets):
        raise ValueError("duplicate input combination in test split")
    if set(train_sets).intersection(test_sets):
        raise ValueError("train and test splits overlap by underlying combination")
    return LoadedDataset(metadata, train, test)


def _run_selftest() -> None:
    config = DatasetConfig(
        n=9,
        m=4,
        train_percent=None,
        train_count=20,
        modulus=3,
        seed=17,
        n_test=15,
    )
    train_ranks, test_ranks, total = sortdata.split_ranks(config)
    first = list(iter_examples(train_ranks, config, "train"))
    second = list(iter_examples(train_ranks, config, "train"))
    assert first == second
    assert all(len(set(example.inputs)) == config.m for example in first)
    assert all(tuple(sorted(example.inputs)) == example.asc for example in first)
    with tempfile.TemporaryDirectory() as temp:
        write_dataset(temp, config, train_ranks, test_ranks, total)
        loaded = load_dataset(temp)
        assert list(loaded.train.examples) == first
        assert len(loaded.test.examples) == len(test_ranks)
        assert set(tuple(sorted(row.inputs)) for row in loaded.train.examples).isdisjoint(
            tuple(sorted(row.inputs)) for row in loaded.test.examples
        )
    print("selftest: PASS (random order, distinct values, targets, split, roundtrip)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n", type=int, default=20, help="universe is [0, n)")
    parser.add_argument("--m", type=int, default=4, help="fixed input sequence length")
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument("--train-count", type=int)
    sizing.add_argument("--train-percent", type=float)
    parser.add_argument(
        "--split-strategy", choices=("random", "relation-complete"), default="random"
    )
    parser.add_argument("--modulus", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--enumerate-limit", type=int, default=5_000_000)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--preview", type=int, nargs="?", const=5, default=5, metavar="N"
    )
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.selftest:
        _run_selftest()
        return 0
    if args.preview < 0:
        parser.error("--preview must be nonnegative")
    train_percent = (
        30.0
        if args.train_count is None and args.train_percent is None
        else args.train_percent
    )
    config = DatasetConfig(
        n=args.n,
        m=args.m,
        train_percent=train_percent,
        train_count=args.train_count,
        split_strategy=args.split_strategy,
        modulus=args.modulus,
        seed=args.seed,
        n_test=args.n_test,
        enumerate_limit=args.enumerate_limit,
    )
    try:
        train_ranks, test_ranks, total = sortdata.split_ranks(config)
        coverage = (
            write_dataset(args.out, config, train_ranks, test_ranks, total)
            if args.out is not None
            else sortdata.coverage_report(train_ranks, test_ranks, config)
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"total={total:,} train={len(train_ranks):,} test={len(test_ranks):,} "
        f"strategy={config.split_strategy} input=random-order distinct=yes length={config.m}"
    )
    if args.out is not None:
        print(f"wrote {args.out / 'train.txt'}, {args.out / 'test.txt'}, and metadata.json")
    for task in sortdata.TASKS:
        adjacent = coverage["adjacent_pairs"][task]
        strata = coverage["test_strata"][task]
        print(
            f"{task} adjacent={adjacent['covered']:,}/{adjacent['possible']:,} "
            f"identifiable={'yes' if coverage['order_identifiable'][task] else 'no'} | "
            f"test direct={strata['direct']:,} transitive={strata['transitive']:,} "
            f"unresolved={strata['unresolved']:,}"
        )
    if args.preview:
        preview = list(iter_examples(train_ranks[: args.preview], config, "train"))
        if len(preview) < args.preview:
            preview.extend(
                iter_examples(
                    test_ranks[: args.preview - len(preview)], config, "test"
                )
            )
        for example in preview:
            print(_format_example(example))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
