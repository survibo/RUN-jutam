"""Deterministic datasets for learning several sorting rules.

The public ``load_split`` and ``load_dataset`` functions intentionally return
plain immutable Python data, so callers do not need numpy or torch.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Iterable, Iterator, Sequence


FORMAT_VERSION = 2
DEFAULT_LARGE_TEST = 50_000
MAX_MATERIALIZED_ROWS = 500_000
MAX_MATERIALIZED_CELLS = 5_000_000
MAX_PAIR_COVERAGE_IDS = 1_000_000
_TEST_SEED_XOR = 0x9E3779B97F4A7C15
_BASIS_SEED_XOR = 0xD1B54A32D192ED03
TASKS = ("ascending", "mod", "alternating")


@dataclass(frozen=True)
class SortExample:
    """One canonical input set and its three target sequences."""

    inputs: tuple[int, ...]
    asc: tuple[int, ...]
    mod: tuple[int, ...]
    alt: tuple[int, ...]


@dataclass(frozen=True)
class DatasetConfig:
    n: int = 20
    m: int = 4
    train_percent: float | None = 30.0
    modulus: int = 3
    seed: int = 0
    n_test: int = -1
    enumerate_limit: int = 5_000_000
    train_count: int | None = None
    split_strategy: str = "random"


@dataclass(frozen=True)
class LoadedSplit:
    config: dict[str, object]
    split: str
    examples: tuple[SortExample, ...]


@dataclass(frozen=True)
class LoadedDataset:
    metadata: dict[str, object]
    train: LoadedSplit
    test: LoadedSplit


@dataclass(frozen=True)
class TrainComparisonInformation:
    """Direct and transitively inferable ordered comparisons as integer bitsets."""

    n: int
    task: str
    order: tuple[int, ...]
    positions: tuple[int, ...]
    direct: tuple[int, ...]
    reachable: tuple[int, ...]


def combination_rank(values: Sequence[int], n: int) -> int:
    """Return the zero-based lexicographic rank of a combination."""
    m = len(values)
    if n < 0 or m > n or any(type(x) is not int for x in values):
        raise ValueError("values must be an integer combination in [0, n)")
    if any(x < 0 or x >= n for x in values) or any(
        values[i] >= values[i + 1] for i in range(m - 1)
    ):
        raise ValueError("values must be strictly increasing and in [0, n)")

    rank = 0
    start = 0
    for i, value in enumerate(values):
        remaining = m - i
        rank += math.comb(n - start, remaining) - math.comb(n - value, remaining)
        start = value + 1
    return rank


def combination_unrank(rank: int, n: int, m: int) -> tuple[int, ...]:
    """Return the lexicographically ranked combination without enumeration."""
    if n < 0 or m < 0 or m > n:
        raise ValueError("require 0 <= m <= n")
    total = math.comb(n, m)
    if rank < 0 or rank >= total:
        raise ValueError(f"rank must be in [0, {total})")

    result: list[int] = []
    start = 0
    for i in range(m):
        remaining = m - i
        base = math.comb(n - start, remaining)
        low, high = start, n - remaining
        while low < high:
            mid = (low + high + 1) // 2
            skipped = base - math.comb(n - mid, remaining)
            if skipped <= rank:
                low = mid
            else:
                high = mid - 1
        value = low
        rank -= base - math.comb(n - value, remaining)
        result.append(value)
        start = value + 1
    return tuple(result)


def make_example(values: Sequence[int], modulus: int) -> SortExample:
    """Build all targets for a canonical (sorted, distinct) input set."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    asc = tuple(values)
    if any(asc[i] >= asc[i + 1] for i in range(len(asc) - 1)):
        raise ValueError("inputs must be strictly increasing")
    mod = tuple(sorted(asc, key=lambda x: (x % modulus, x)))
    alt_list: list[int] = []
    low, high = 0, len(asc) - 1
    while low <= high:
        alt_list.append(asc[low])
        if low != high:
            alt_list.append(asc[high])
        low += 1
        high -= 1
    return SortExample(asc, asc, mod, tuple(alt_list))


def _sample_unique(total: int, count: int, rng: random.Random) -> list[int]:
    """Floyd sampling, using O(count) memory even when total exceeds sys.maxsize."""
    if count < 0 or count > total:
        raise ValueError("sample size is outside the population")
    selected: set[int] = set()
    result: list[int] = []
    j = total - count
    while j < total:
        candidate = rng.randrange(j + 1)
        chosen = j if candidate in selected else candidate
        selected.add(chosen)
        result.append(chosen)
        j += 1
    return result


def _kth_not_excluded(index: int, excluded: Sequence[int], total: int) -> int:
    """Map an index in a complement to its rank without building the complement."""
    low, high = index, index + len(excluded)
    if high >= total:
        high = total - 1
    while low < high:
        mid = (low + high) // 2
        present = mid + 1 - bisect.bisect_right(excluded, mid)
        if present >= index + 1:
            high = mid
        else:
            low = mid + 1
    return low


def _resolved_train_count(config: DatasetConfig, total: int) -> int:
    if config.train_count is not None:
        train_count = config.train_count
    else:
        assert config.train_percent is not None
        train_count = int(
            (
                Decimal(total)
                * Decimal(str(config.train_percent))
                / Decimal(100)
            ).to_integral_value(rounding=ROUND_HALF_EVEN)
        )
    if train_count > total:
        raise ValueError(
            f"train-count={train_count} exceeds the {total} available combinations"
        )
    return train_count


def _task_order(config: DatasetConfig, task: str) -> tuple[int, ...]:
    if task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)}")
    if task == "mod":
        return tuple(sorted(range(config.n), key=lambda value: (value % config.modulus, value)))
    return tuple(range(config.n))


def _relation_complete_basis(config: DatasetConfig) -> list[int]:
    if config.n == 1:
        return [0]
    if config.m < 2:
        raise ValueError("relation-complete requires m >= 2 when n > 1")

    numeric_order = _task_order(config, "ascending")
    mod_order = _task_order(config, "mod")
    requirements: dict[int, tuple[int, int]] = {}
    for order in (numeric_order, mod_order):
        for left, right in zip(order, order[1:]):
            pair = (min(left, right), max(left, right))
            requirements.setdefault(_pair_id(*pair, config.n), pair)
    required_pairs = tuple(requirements)
    required_bits = {pair: 1 << index for index, pair in enumerate(required_pairs)}
    candidates = tuple(
        dict.fromkeys(
            tuple(sorted(order[start : start + config.m]))
            for order in (numeric_order, mod_order)
            for start in range(config.n - config.m + 1)
        )
    )
    candidate_masks: list[int] = []
    for values in candidates:
        value_set = set(values)
        mask = 0
        for pair, bit in required_bits.items():
            left, right = requirements[pair]
            if left in value_set and right in value_set:
                mask |= bit
        candidate_masks.append(mask)

    priorities = list(range(len(candidates)))
    random.Random(config.seed ^ _BASIS_SEED_XOR).shuffle(priorities)
    priority = {candidate: order for order, candidate in enumerate(priorities)}
    uncovered = (1 << len(required_pairs)) - 1
    frequencies = [0] * config.n
    selected: list[int] = []
    available = set(range(len(candidates)))

    while uncovered:
        best = min(
            available,
            key=lambda candidate: (
                -(candidate_masks[candidate] & uncovered).bit_count(),
                sum(frequencies[value] for value in candidates[candidate]),
                priority[candidate],
            ),
        )
        if not candidate_masks[best] & uncovered:
            raise ValueError("relation-complete could not cover all numeric and mod requirements")
        selected.append(combination_rank(candidates[best], config.n))
        for value in candidates[best]:
            frequencies[value] += 1
        uncovered &= ~candidate_masks[best]
        available.remove(best)
    return selected


def _sample_complement(total: int, count: int, excluded: Sequence[int], seed: int) -> list[int]:
    sorted_excluded = sorted(excluded)
    indices = _sample_unique(total - len(sorted_excluded), count, random.Random(seed))
    return [_kth_not_excluded(index, sorted_excluded, total) for index in indices]


def split_ranks(config: DatasetConfig) -> tuple[list[int], list[int], int]:
    """Create exact-size, deterministic, disjoint train and test rank lists."""
    _validate_config(config)
    total = math.comb(config.n, config.m)
    train_count = _resolved_train_count(config, total)
    remaining = total - train_count
    if config.n_test == -1:
        manageable = min(config.enumerate_limit, MAX_MATERIALIZED_ROWS)
        test_count = remaining if remaining <= manageable else min(remaining, DEFAULT_LARGE_TEST)
    else:
        test_count = config.n_test
        if test_count > remaining:
            raise ValueError(
                f"n-test={test_count} exceeds the {remaining} non-training combinations"
            )
    if train_count + test_count > MAX_MATERIALIZED_ROWS:
        raise ValueError(
            f"requested {train_count + test_count:,} rows; safety limit is "
            f"{MAX_MATERIALIZED_ROWS:,}. Reduce training size/--n-test or the universe."
        )
    cells = (train_count + test_count) * config.m
    if cells > MAX_MATERIALIZED_CELLS:
        raise ValueError(
            f"requested {cells:,} integer cells; safety limit is "
            f"{MAX_MATERIALIZED_CELLS:,}. Reduce rows or --m."
        )

    if config.split_strategy == "random":
        if _should_enumerate(total, config):
            ranks = list(range(total))
            random.Random(config.seed).shuffle(ranks)
            return ranks[:train_count], ranks[train_count : train_count + test_count], total
        train = _sample_unique(total, train_count, random.Random(config.seed))
    else:
        if test_count == 0:
            raise ValueError("relation-complete requires at least one test example")
        train = _relation_complete_basis(config)
        if train_count < len(train):
            raise ValueError(
                f"relation-complete requires at least {len(train)} training examples "
                f"for n={config.n}, m={config.m}; requested {train_count}"
            )
        train.extend(
            _sample_complement(total, train_count - len(train), train, config.seed)
        )
    test = _sample_complement(total, test_count, train, config.seed ^ _TEST_SEED_XOR)
    return train, test, total


def compute_train_comparison_information(
    train_ranks: Iterable[int], config: DatasetConfig, task: str = "ascending"
) -> TrainComparisonInformation:
    """Compute comparison exposure and closure for one dataset task order."""
    order = _task_order(config, task)
    positions_list = [0] * config.n
    for position, value in enumerate(order):
        positions_list[value] = position
    positions = tuple(positions_list)
    direct = [0] * config.n
    for rank in train_ranks:
        values = combination_unrank(rank, config.n, config.m)
        ordered_positions = sorted(positions[value] for value in values)
        for left_index, left in enumerate(ordered_positions):
            for right in ordered_positions[left_index + 1 :]:
                direct[left] |= 1 << right

    reachable = direct.copy()
    for left in range(config.n - 1, -1, -1):
        pending = direct[left]
        while pending:
            bit = pending & -pending
            right = bit.bit_length() - 1
            reachable[left] |= reachable[right]
            pending ^= bit
    return TrainComparisonInformation(
        config.n, task, order, positions, tuple(direct), tuple(reachable)
    )


def classify_test_example(
    example: SortExample | Sequence[int], information: TrainComparisonInformation
) -> str:
    """Classify an example or its input values as direct, transitive, or unresolved."""
    values = example.inputs if isinstance(example, SortExample) else example
    combination_rank(values, information.n)
    ordered_positions = sorted(information.positions[value] for value in values)
    all_direct = True
    all_reachable = True
    for left_index, left in enumerate(ordered_positions):
        for right in ordered_positions[left_index + 1 :]:
            bit = 1 << right
            all_direct &= bool(information.direct[left] & bit)
            all_reachable &= bool(information.reachable[left] & bit)
    if all_direct:
        return "direct"
    if all_reachable:
        return "transitive"
    return "unresolved"


def classify_test_rank(
    rank: int, config: DatasetConfig, information: TrainComparisonInformation
) -> str:
    """Classify one lexicographic combination rank."""
    if information.n != config.n:
        raise ValueError("comparison information and config have different n values")
    return classify_test_example(
        combination_unrank(rank, config.n, config.m), information
    )


def classify_test_ranks(
    test_ranks: Iterable[int],
    config: DatasetConfig,
    information: TrainComparisonInformation,
) -> list[str]:
    """Classify test ranks in input order."""
    return [classify_test_rank(rank, config, information) for rank in test_ranks]


def iter_examples(ranks: Iterable[int], config: DatasetConfig) -> Iterator[SortExample]:
    for rank in ranks:
        yield make_example(combination_unrank(rank, config.n, config.m), config.modulus)


def _frequency_summary(counts: Iterable[int], possible: int, observations: int) -> dict[str, float | int]:
    nonzero = list(counts)
    sum_squares = sum(value * value for value in nonzero)
    mean = observations / possible if possible else 0.0
    variance = sum_squares / possible - mean * mean if possible else 0.0
    return {
        "covered": len(nonzero),
        "possible": possible,
        "coverage_percent": 100.0 * len(nonzero) / possible if possible else 100.0,
        "frequency_min": 0 if len(nonzero) < possible else min(nonzero, default=0),
        "frequency_max": max(nonzero, default=0),
        "frequency_mean": mean,
        "frequency_std": math.sqrt(max(0.0, variance)),
    }


def _pair_id(first: int, second: int, n: int) -> int:
    return first * (2 * n - first - 1) // 2 + second - first - 1


def coverage_report(
    train_ranks: Sequence[int], test_ranks: Sequence[int], config: DatasetConfig
) -> dict[str, object]:
    """Summarize element and pair exposure without storing per-item details."""
    element_counts: collections.Counter[int] = collections.Counter()
    adjacent_pairs = {
        task: tuple(
            (min(left, right), max(left, right))
            for left, right in zip(
                _task_order(config, task), _task_order(config, task)[1:]
            )
        )
        for task in TASKS
    }
    adjacent_counts = {task: collections.Counter() for task in TASKS}
    possible_pairs = math.comb(config.n, 2)
    potential_train_pairs = len(train_ranks) * math.comb(config.m, 2)
    potential_test_pairs = len(test_ranks) * math.comb(config.m, 2)
    pair_analysis_enabled = sum((
        min(possible_pairs, potential_train_pairs),
        min(possible_pairs, potential_test_pairs),
    )) <= MAX_PAIR_COVERAGE_IDS
    pair_counts: collections.Counter[int] | None = (
        collections.Counter() if pair_analysis_enabled else None
    )

    for rank in train_ranks:
        values = combination_unrank(rank, config.n, config.m)
        element_counts.update(values)
        value_set = set(values)
        for task in TASKS:
            adjacent_counts[task].update(
                _pair_id(left, right, config.n)
                for left, right in adjacent_pairs[task]
                if left in value_set and right in value_set
            )
        if pair_counts is not None:
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    pair_counts[_pair_id(values[left], values[right], config.n)] += 1

    comparison_information = {
        task: compute_train_comparison_information(train_ranks, config, task)
        for task in TASKS
    }
    strata = {
        task: collections.Counter(
            classify_test_ranks(test_ranks, config, comparison_information[task])
        )
        for task in TASKS
    }
    all_values_mask = (1 << config.n) - 1
    higher_value_masks = tuple(
        all_values_mask & ~((1 << (left + 1)) - 1) for left in range(config.n)
    )
    order_identifiable = {
        task: all(
            comparison_information[task].reachable[left] & higher_value_masks[left]
            == higher_value_masks[left]
            for left in range(config.n)
        )
        for task in TASKS
    }
    report: dict[str, object] = {
        "elements": _frequency_summary(
            element_counts.values(), config.n, len(train_ranks) * config.m
        ),
        "adjacent_pairs": {
            task: _frequency_summary(
                adjacent_counts[task].values(),
                max(0, config.n - 1),
                sum(adjacent_counts[task].values()),
            )
            for task in TASKS
        },
        "order_identifiable": order_identifiable,
        "test_strata": {
            task: {
                name: strata[task].get(name, 0)
                for name in ("direct", "transitive", "unresolved")
            }
            for task in TASKS
        },
    }
    if pair_counts is None:
        report["pairs"] = {
            "status": "skipped",
            "reason": (
                f"exact pair analysis would track more than "
                f"{MAX_PAIR_COVERAGE_IDS:,} pair ids"
            ),
            "possible": possible_pairs,
            "potential_train_occurrences": potential_train_pairs,
            "potential_test_occurrences": potential_test_pairs,
        }
    else:
        pair_summary = _frequency_summary(
            pair_counts.values(), possible_pairs, potential_train_pairs
        )
        test_pair_occurrences = 0
        test_pair_seen = 0
        test_unique_pairs: set[int] = set()
        for rank in test_ranks:
            values = combination_unrank(rank, config.n, config.m)
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    pair = _pair_id(values[left], values[right], config.n)
                    test_pair_occurrences += 1
                    test_unique_pairs.add(pair)
                    if pair in pair_counts:
                        test_pair_seen += 1
        test_seen_unique_count = sum(pair in pair_counts for pair in test_unique_pairs)
        pair_summary.update(
            status="complete",
            test_occurrences=test_pair_occurrences,
            test_occurrences_seen_in_train=test_pair_seen,
            test_occurrences_seen_percent=(
                100.0 * test_pair_seen / test_pair_occurrences
                if test_pair_occurrences else 100.0
            ),
            test_unique_pairs=len(test_unique_pairs),
            test_unique_pairs_seen_in_train=test_seen_unique_count,
            test_unique_pairs_seen_percent=(
                100.0 * test_seen_unique_count / len(test_unique_pairs)
                if test_unique_pairs else 100.0
            ),
        )
        report["pairs"] = pair_summary
    return report


def _validate_config(config: DatasetConfig) -> None:
    if config.n < 1:
        raise ValueError("n must be positive")
    if config.m < 1 or config.m > config.n:
        raise ValueError("m must satisfy 1 <= m <= n")
    if (config.train_percent is None) == (config.train_count is None):
        raise ValueError("exactly one of train-percent and train-count must be set")
    if config.train_percent is not None and (
        not math.isfinite(config.train_percent) or not 0 <= config.train_percent <= 100
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


def _should_enumerate(total: int, config: DatasetConfig) -> bool:
    return total <= min(config.enumerate_limit, MAX_MATERIALIZED_ROWS)


def _format_example(example: SortExample) -> str:
    fields = lambda values: " ".join(map(str, values))
    return (
        f"{fields(example.inputs)} -> asc: {fields(example.asc)} | "
        f"mod: {fields(example.mod)} | alt: {fields(example.alt)}"
    )


def _header(config: DatasetConfig, split: str, count: int, total: int) -> str:
    payload = asdict(config)
    payload.update(
        format_version=FORMAT_VERSION,
        split=split,
        count=count,
        total=total,
        train_sizing_mode="count" if config.train_count is not None else "percent",
    )
    return "# sortdata-config: " + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_split(
    path: Path, config: DatasetConfig, split: str, ranks: Sequence[int], total: int
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_header(config, split, len(ranks), total) + "\n")
        handle.write("# input -> asc: ... | mod: ... | alt: ...\n")
        for example in iter_examples(ranks, config):
            handle.write(_format_example(example) + "\n")


def write_dataset(
    out: str | Path,
    config: DatasetConfig,
    train_ranks: Sequence[int],
    test_ranks: Sequence[int],
    total: int,
) -> dict[str, object]:
    """Write train.txt, test.txt, and metadata.json to ``out``."""
    out_path = Path(out)
    if out_path.exists() and not out_path.is_dir():
        raise ValueError(f"output path is not a directory: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    _write_split(out_path / "train.txt", config, "train", train_ranks, total)
    _write_split(out_path / "test.txt", config, "test", test_ranks, total)
    coverage = coverage_report(train_ranks, test_ranks, config)
    sizing_mode = "count" if config.train_count is not None else "percent"
    metadata = {
        "format_version": FORMAT_VERSION,
        "config": asdict(config),
        "split_strategy": config.split_strategy,
        "train_sizing": {
            "mode": sizing_mode,
            "requested": (
                config.train_count if config.train_count is not None else config.train_percent
            ),
            "effective_count": len(train_ranks),
        },
        "total_combinations": total,
        "train_count": len(train_ranks),
        "test_count": len(test_ranks),
        "test_is_all_remaining": len(test_ranks) == total - len(train_ranks),
        "files": {"train": "train.txt", "test": "test.txt"},
        "coverage": coverage,
    }
    with (out_path / "metadata.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return coverage


def _parse_values(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.strip().split())


def _parse_example(line: str) -> SortExample:
    try:
        input_text, targets = line.split(" -> ", 1)
        sections = targets.split(" | ")
        labelled = dict(section.split(": ", 1) for section in sections)
        example = SortExample(
            _parse_values(input_text),
            _parse_values(labelled["asc"]),
            _parse_values(labelled["mod"]),
            _parse_values(labelled["alt"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid sortdata row: {line!r}") from exc
    return example


def load_split(path: str | Path, *, validate: bool = True) -> LoadedSplit:
    """Load one generated text split and optionally verify every target."""
    source = Path(path)
    config: dict[str, object] | None = None
    examples: list[SortExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if line.startswith("# sortdata-config: "):
                config = json.loads(line.removeprefix("# sortdata-config: "))
            elif line and not line.startswith("#"):
                examples.append(_parse_example(line))
    if config is None:
        raise ValueError(f"missing sortdata config header in {source}")
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported format version in {source}")
    required_config_keys = set(asdict(DatasetConfig()))
    if not required_config_keys.issubset(config):
        missing = sorted(required_config_keys.difference(config))
        raise ValueError(f"missing config fields in {source}: {', '.join(missing)}")
    parsed_config = DatasetConfig(
        **{key: config[key] for key in required_config_keys}
    )
    _validate_config(parsed_config)
    expected_sizing_mode = "count" if config["train_count"] is not None else "percent"
    if config.get("train_sizing_mode") != expected_sizing_mode:
        raise ValueError(f"invalid train sizing mode in {source}")
    modulus = int(config["modulus"])
    m = int(config["m"])
    if int(config["count"]) != len(examples):
        raise ValueError(f"row count does not match header in {source}")
    if validate:
        for example in examples:
            if len(example.inputs) != m or make_example(example.inputs, modulus) != example:
                raise ValueError(f"invalid targets or input in {source}: {example.inputs}")
    return LoadedSplit(config, str(config["split"]), tuple(examples))


def load_dataset(path: str | Path, *, validate: bool = True) -> LoadedDataset:
    """Load a directory produced by :func:`write_dataset`."""
    directory = Path(path)
    with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported metadata format version")
    train = load_split(directory / metadata["files"]["train"], validate=validate)
    test = load_split(directory / metadata["files"]["test"], validate=validate)
    if train.split != "train" or test.split != "test":
        raise ValueError("dataset split labels must be train and test")
    metadata_config = metadata.get("config")
    if not isinstance(metadata_config, dict):
        raise ValueError("metadata config must be an object")
    config_keys = tuple(asdict(DatasetConfig()).keys())
    for split in (train, test):
        split_config = {key: split.config.get(key) for key in config_keys}
        if split_config != metadata_config:
            raise ValueError(f"{split.split} header does not match metadata config")
    if metadata.get("train_count") != len(train.examples) or metadata.get(
        "test_count"
    ) != len(test.examples):
        raise ValueError("metadata row counts do not match split files")
    expected_mode = "count" if metadata_config.get("train_count") is not None else "percent"
    expected_sizing = {
        "mode": expected_mode,
        "requested": metadata_config.get(
            "train_count" if expected_mode == "count" else "train_percent"
        ),
        "effective_count": len(train.examples),
    }
    if metadata.get("split_strategy") != metadata_config.get("split_strategy"):
        raise ValueError("metadata split strategy does not match config")
    if metadata.get("train_sizing") != expected_sizing:
        raise ValueError("metadata train sizing does not match config and files")
    coverage = metadata.get("coverage", {})
    required_coverage = {
        "elements", "pairs", "adjacent_pairs", "order_identifiable", "test_strata"
    }
    if not isinstance(coverage, dict) or not required_coverage.issubset(coverage):
        raise ValueError("metadata coverage is missing required fields")
    adjacent = coverage["adjacent_pairs"]
    identifiable = coverage["order_identifiable"]
    strata = coverage["test_strata"]
    if not all(
        isinstance(section, dict) and set(section) == set(TASKS)
        for section in (adjacent, identifiable, strata)
    ):
        raise ValueError("metadata task coverage must contain all dataset tasks")
    if not all(type(identifiable[task]) is bool for task in TASKS):
        raise ValueError("metadata task identifiability values must be booleans")
    for task in TASKS:
        task_strata = strata[task]
        if (
            not isinstance(task_strata, dict)
            or set(task_strata) != {"direct", "transitive", "unresolved"}
            or any(type(value) is not int or value < 0 for value in task_strata.values())
            or sum(task_strata.values()) != len(test.examples)
        ):
            raise ValueError(f"metadata {task} test strata do not match test row count")
        if not isinstance(adjacent[task], dict) or not {
            "covered", "possible", "coverage_percent"
        }.issubset(adjacent[task]):
            raise ValueError(f"metadata {task} adjacent-pair coverage is invalid")
    train_inputs = [example.inputs for example in train.examples]
    test_inputs = [example.inputs for example in test.examples]
    if len(set(train_inputs)) != len(train_inputs):
        raise ValueError("duplicate input combination in train split")
    if len(set(test_inputs)) != len(test_inputs):
        raise ValueError("duplicate input combination in test split")
    if set(train_inputs).intersection(test_inputs):
        raise ValueError("train and test splits overlap")
    return LoadedDataset(metadata, train, test)


def _run_selftest() -> None:
    # Exhaustive small-space rank roundtrips also check lexicographic uniqueness.
    for n in range(1, 10):
        for m in range(1, n + 1):
            for rank in range(math.comb(n, m)):
                values = combination_unrank(rank, n, m)
                assert combination_rank(values, n) == rank

    example = make_example((0, 3, 7, 9), 4)
    assert example.asc == (0, 3, 7, 9)
    assert example.mod == (0, 9, 3, 7)
    assert example.alt == (0, 9, 3, 7)
    assert make_example((1, 2, 3, 4, 5), 3).alt == (1, 5, 2, 4, 3)

    orientation_config = DatasetConfig(n=6, m=2, modulus=3)
    orientation_rank = combination_rank((1, 3), orientation_config.n)
    mod_info = compute_train_comparison_information(
        [orientation_rank], orientation_config, "mod"
    )
    assert mod_info.order == (0, 3, 1, 4, 2, 5)
    assert mod_info.positions[3] < mod_info.positions[1]
    assert mod_info.direct[mod_info.positions[3]] & (1 << mod_info.positions[1])
    assert compute_train_comparison_information(
        [orientation_rank], orientation_config, "alternating"
    ).order == tuple(range(orientation_config.n))

    small = DatasetConfig(n=12, m=3, train_percent=31.0, seed=73)
    first = split_ranks(small)
    second = split_ranks(small)
    assert first == second
    assert len(first[0]) == round(math.comb(12, 3) * 0.31)
    assert set(first[0]).isdisjoint(first[1])

    large = DatasetConfig(
        n=100, m=8, train_percent=0.000001, seed=19, n_test=1000, enumerate_limit=10
    )
    train, test, total = split_ranks(large)
    assert total > 10 and set(train).isdisjoint(test)
    assert len(test) == len(set(test)) == 1000
    assert split_ranks(large) == (train, test, total)

    exact = DatasetConfig(n=12, m=3, train_percent=None, train_count=17, seed=73)
    exact_train, exact_test, _ = split_ranks(exact)
    assert len(exact_train) == 17 and set(exact_train).isdisjoint(exact_test)

    relation_template = DatasetConfig(
        n=12,
        m=3,
        train_percent=None,
        train_count=1,
        split_strategy="relation-complete",
        seed=73,
        n_test=40,
    )
    basis_count = len(_relation_complete_basis(relation_template))
    relation = DatasetConfig(
        n=relation_template.n,
        m=relation_template.m,
        train_percent=None,
        train_count=basis_count + 2,
        split_strategy="relation-complete",
        seed=relation_template.seed,
        n_test=relation_template.n_test,
    )
    relation_split = split_ranks(relation)
    assert split_ranks(relation) == relation_split
    relation_train, relation_test, _ = relation_split
    assert len(relation_train) == basis_count + 2
    for task in TASKS:
        relation_info = compute_train_comparison_information(
            relation_train, relation, task
        )
        assert all(
            relation_info.direct[position] & (1 << (position + 1))
            for position in range(relation.n - 1)
        )
        relation_strata = classify_test_ranks(
            relation_test, relation, relation_info
        )
        assert "unresolved" not in relation_strata
        first_test = next(iter_examples(relation_test[:1], relation))
        assert classify_test_example(first_test, relation_info) == relation_strata[0]
    relation_coverage = coverage_report(relation_train, relation_test, relation)
    for task in TASKS:
        assert relation_coverage["adjacent_pairs"][task]["covered"] == relation.n - 1
        assert relation_coverage["order_identifiable"][task] is True
        assert relation_coverage["test_strata"][task]["unresolved"] == 0

    random_sparse = DatasetConfig(
        n=8, m=2, train_percent=None, train_count=1, seed=4, n_test=5
    )
    sparse_train, sparse_test, _ = split_ranks(random_sparse)
    sparse_info = compute_train_comparison_information(sparse_train, random_sparse)
    assert "unresolved" in classify_test_ranks(sparse_test, random_sparse, sparse_info)

    try:
        split_ranks(
            DatasetConfig(
                n=12,
                m=3,
                train_percent=None,
                train_count=basis_count - 1,
                split_strategy="relation-complete",
                seed=relation_template.seed,
            )
        )
    except ValueError as exc:
        assert "requires at least" in str(exc)
    else:
        raise AssertionError("too-small relation-complete split did not fail")

    with tempfile.TemporaryDirectory() as temp:
        train, test, total = split_ranks(small)
        write_dataset(temp, small, train, test, total)
        loaded = load_dataset(temp)
        assert len(loaded.train.examples) == len(train)
        assert len(loaded.test.examples) == len(test)
        assert loaded.train.examples[0] == next(iter_examples(train[:1], small))
        coverage = loaded.metadata["coverage"]
        assert coverage["elements"]["covered"] == small.n
        assert coverage["pairs"]["status"] == "complete"
        assert 0 <= coverage["pairs"]["test_occurrences_seen_percent"] <= 100
    print(
        "selftest: PASS (ranking, targets, exact/random/relation splits, "
        "classification, large sampling, file roundtrip)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n", type=int, default=20, help="universe is [0, n)")
    parser.add_argument("--m", type=int, default=4, help="input set size")
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
        "--preview", type=int, nargs="?", const=5, default=5, metavar="N",
        help="show N rows (bare --preview means 5; 0 disables)",
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
    train_percent = 30.0 if args.train_count is None and args.train_percent is None else args.train_percent
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
        train, test, total = split_ranks(config)
        if args.out is not None:
            coverage = write_dataset(args.out, config, train, test, total)
        else:
            coverage = coverage_report(train, test, config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    enumerated = config.split_strategy == "random" and _should_enumerate(total, config)
    print(
        f"total={total:,} train={len(train):,} test={len(test):,} "
        f"strategy={config.split_strategy} "
        f"sizing={'count' if config.train_count is not None else 'percent'} "
        f"mode={'enumerated' if enumerated else 'sampled'}"
    )
    if args.out is not None:
        print(f"wrote {args.out / 'train.txt'}, {args.out / 'test.txt'}, and metadata.json")
    elements = coverage["elements"]
    pairs = coverage["pairs"]
    adjacent = coverage["adjacent_pairs"]
    strata = coverage["test_strata"]
    for task in TASKS:
        task_adjacent = adjacent[task]
        task_strata = strata[task]
        print(
            f"{task} adjacent-pair coverage: "
            f"{task_adjacent['covered']:,}/{task_adjacent['possible']:,} "
            f"({task_adjacent['coverage_percent']:.2f}%) | order identifiable: "
            f"{'yes' if coverage['order_identifiable'][task] else 'no'}"
        )
        print(
            f"{task} test strata: direct={task_strata['direct']:,} "
            f"transitive={task_strata['transitive']:,} "
            f"unresolved={task_strata['unresolved']:,}"
        )
    if args.out is not None:
        print(
            f"element coverage: {elements['covered']:,}/{elements['possible']:,} "
            f"({elements['coverage_percent']:.2f}%)"
        )
        if pairs["status"] == "complete":
            print(
                f"pair coverage: {pairs['covered']:,}/{pairs['possible']:,} "
                f"({pairs['coverage_percent']:.2f}%) | test pair occurrences seen: "
                f"{pairs['test_occurrences_seen_percent']:.2f}%"
            )
        else:
            print(f"pair coverage: skipped ({pairs['reason']})")
    if args.preview:
        preview_ranks = train[: args.preview]
        if len(preview_ranks) < args.preview:
            preview_ranks += test[: args.preview - len(preview_ranks)]
        for example in iter_examples(preview_ranks, config):
            print(_format_example(example))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
