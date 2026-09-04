"""Jointly learn entity-value facts and transfer ID sorting to OOD entities.

One GPT-2-style causal Transformer learns both sequence types::

    [BOS, ATOM, ATTR, entity, SEP, value, EOS]
    [BOS, SORT, ATTR, entities..., SEP, value-sorted entities..., EOS]

All entities occur in atomic training. Sorting supervision uses ID entities
only; validation uses unseen ID combinations and test uses OOD entities only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import tempfile
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100
CSV_COLUMNS = (
    "step", "epoch", "examples_seen", "lr", "weight_norm",
    "train_loss", "train_token_acc", "train_exact_acc",
    "validation_loss", "validation_token_acc", "validation_exact_acc",
    "test_loss", "test_token_acc", "test_exact_acc",
    "atomic_train_loss", "atomic_train_token_acc", "atomic_train_exact_acc",
    "atomic_id_loss", "atomic_id_token_acc", "atomic_id_exact_acc",
    "atomic_ood_loss", "atomic_ood_token_acc", "atomic_ood_exact_acc",
    "sorting_train_loss", "sorting_train_token_acc", "sorting_train_exact_acc",
    "id_validation_sorting_loss", "id_validation_sorting_token_acc",
    "id_validation_sorting_exact_acc", "ood_test_sorting_loss",
    "ood_test_sorting_token_acc", "ood_test_sorting_exact_acc",
    "atomic_id_invalid_token_rate", "atomic_id_early_eos_rate",
    "atomic_ood_invalid_token_rate", "atomic_ood_early_eos_rate",
    "sorting_train_invalid_token_rate", "sorting_train_early_eos_rate",
    "sorting_train_duplicate_rate", "sorting_train_not_in_input_rate",
    "id_validation_invalid_token_rate", "id_validation_early_eos_rate",
    "id_validation_duplicate_rate", "id_validation_not_in_input_rate",
    "ood_test_invalid_token_rate", "ood_test_early_eos_rate",
    "ood_test_duplicate_rate", "ood_test_not_in_input_rate",
    "training_mode", "configured_atomic_fraction", "effective_atomic_fraction",
    "dynamic_input_permutation", "elapsed_seconds", "train_eval_count",
    "validation_eval_count", "test_eval_count", "atomic_id_eval_count",
    "atomic_ood_eval_count", "run_signature_sha256",
)


# 1. Vocabulary ----------------------------------------------------------------


@dataclass(frozen=True)
class TaskVocabulary:
    n_entities: int
    value_min: int
    value_max: int

    def __post_init__(self) -> None:
        if self.n_entities < 1 or self.value_max < self.value_min:
            raise ValueError("invalid entity/value vocabulary")

    @property
    def n_values(self) -> int:
        return self.value_max - self.value_min + 1

    @property
    def value_offset(self) -> int:
        return self.n_entities

    @property
    def bos_id(self) -> int:
        return self.n_entities + self.n_values

    @property
    def sep_id(self) -> int:
        return self.bos_id + 1

    @property
    def eos_id(self) -> int:
        return self.bos_id + 2

    @property
    def pad_id(self) -> int:
        return self.bos_id + 3

    @property
    def atom_id(self) -> int:
        return self.bos_id + 4

    @property
    def sort_id(self) -> int:
        return self.bos_id + 5

    @property
    def attr_id(self) -> int:
        return self.bos_id + 6

    @property
    def size(self) -> int:
        return self.bos_id + 7

    def entity_id(self, entity: int) -> int:
        if not 0 <= entity < self.n_entities:
            raise ValueError(f"entity outside [0, {self.n_entities}): {entity}")
        return entity

    def value_id(self, value: int) -> int:
        if not self.value_min <= value <= self.value_max:
            raise ValueError(f"value outside [{self.value_min}, {self.value_max}]: {value}")
        return self.value_offset + value - self.value_min

    def is_entity_id(self, token_id: int) -> bool:
        return 0 <= token_id < self.n_entities

    def is_value_id(self, token_id: int) -> bool:
        return self.value_offset <= token_id < self.bos_id

    def token_name(self, token_id: int) -> str:
        if self.is_entity_id(token_id):
            width = max(4, len(str(self.n_entities - 1)))
            return f"E{token_id:0{width}d}"
        if self.is_value_id(token_id):
            return f"V{self.value_min + token_id - self.value_offset}"
        specials = {
            self.bos_id: "BOS", self.sep_id: "SEP", self.eos_id: "EOS",
            self.pad_id: "PAD", self.atom_id: "ATOM", self.sort_id: "SORT",
            self.attr_id: "ATTR",
        }
        if token_id not in specials:
            raise ValueError(f"token id outside vocabulary: {token_id}")
        return specials[token_id]

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        return [self.token_name(int(token_id)) for token_id in token_ids]


# 2-3. Synthetic KB and fixed entity/combinations splits -----------------------


@dataclass(frozen=True)
class DatasetConfig:
    n_entities: int = 1000
    id_fraction: float = 0.9
    value_min: int = 0
    value_max: int = 100
    set_size: int = 3
    phi: float | None = 3.6
    sorting_train_count: int | None = None
    validation_count: int = 2000
    test_count: int = 5000
    data_seed: int = 0
    combination_enumerate_limit: int = 5_000_000
    max_sampling_attempts: int = 2_000_000
    max_materialized_examples: int = 500_000


@dataclass(frozen=True)
class AtomicExample:
    entity: int
    value: int


@dataclass(frozen=True)
class SortingExample:
    combination: tuple[int, ...]
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


TaskExample = AtomicExample | SortingExample


@dataclass(frozen=True)
class ExperimentData:
    kb: tuple[int, ...]
    id_entities: tuple[int, ...]
    ood_entities: tuple[int, ...]
    atomic_train: tuple[AtomicExample, ...]
    atomic_id: tuple[AtomicExample, ...]
    atomic_ood: tuple[AtomicExample, ...]
    sorting_train: tuple[SortingExample, ...]
    id_validation: tuple[SortingExample, ...]
    ood_test: tuple[SortingExample, ...]
    phi_effective: float


_KB_SEED_XOR = 0xA0761D6478BD642F
_ENTITY_SPLIT_XOR = 0xE7037ED1A0B428DB
_ID_COMBINATION_XOR = 0x8EBC6AF09C88C6E3
_OOD_COMBINATION_XOR = 0x589965CC75374CC3
_TRAIN_INPUT_XOR = 0x1D8E4E27C47D124F
_VALIDATION_INPUT_XOR = 0xEB44ACCAB455D165
_TEST_INPUT_XOR = 0xA4093822299F31D0
_MASK64 = (1 << 64) - 1


def _mixed_seed(seed: int, salt: int, values: Sequence[int] = ()) -> int:
    mixed = (seed ^ salt) & _MASK64
    for value in values:
        mixed ^= (value + 0x9E3779B97F4A7C15 + (mixed << 6) + (mixed >> 2)) & _MASK64
        mixed &= _MASK64
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & _MASK64
    mixed ^= mixed >> 27
    mixed = (mixed * 0x94D049BB133111EB) & _MASK64
    return mixed ^ (mixed >> 31)


def generate_knowledge_base(config: DatasetConfig) -> tuple[int, ...]:
    rng = random.Random(_mixed_seed(config.data_seed, _KB_SEED_XOR))
    return tuple(rng.randint(config.value_min, config.value_max) for _ in range(config.n_entities))


def split_entities(config: DatasetConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    entities = list(range(config.n_entities))
    random.Random(_mixed_seed(config.data_seed, _ENTITY_SPLIT_XOR)).shuffle(entities)
    id_count = round(config.n_entities * config.id_fraction)
    return tuple(sorted(entities[:id_count])), tuple(sorted(entities[id_count:]))


def valid_combination_count(entities: Sequence[int], kb: Sequence[int], set_size: int) -> int:
    """Count combinations selecting at most one entity for each value."""
    result = [0] * (set_size + 1)
    result[0] = 1
    for frequency in Counter(kb[entity] for entity in entities).values():
        for size in range(set_size, 0, -1):
            result[size] += result[size - 1] * frequency
    return result[set_size]


def _sample_valid_combinations(
    entities: Sequence[int], kb: Sequence[int], set_size: int, count: int,
    seed: int, excluded: set[tuple[int, ...]], enumerate_limit: int,
    max_attempts: int,
) -> list[tuple[int, ...]]:
    capacity = valid_combination_count(entities, kb, set_size) - len(excluded)
    if count < 0 or count > capacity:
        raise ValueError(f"requested {count:,} valid combinations; available: {capacity:,}")
    rng = random.Random(seed)
    total = math.comb(len(entities), set_size)
    if total <= enumerate_limit:
        candidates = [
            combination for combination in itertools.combinations(entities, set_size)
            if combination not in excluded
            and len({kb[entity] for entity in combination}) == set_size
        ]
        rng.shuffle(candidates)
        return candidates[:count]
    selected: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    for _ in range(max_attempts):
        if len(result) == count:
            break
        combination = tuple(sorted(rng.sample(entities, set_size)))
        if combination in excluded or combination in selected:
            continue
        if len({kb[entity] for entity in combination}) != set_size:
            continue
        selected.add(combination)
        result.append(combination)
    if len(result) != count:
        raise ValueError(
            f"sampled {len(result):,}/{count:,} valid combinations; raise "
            "--max-sampling-attempts or --combination-enumerate-limit"
        )
    return result


def _make_sorting_example(
    combination: tuple[int, ...], kb: Sequence[int], seed: int, salt: int
) -> SortingExample:
    inputs = list(combination)
    random.Random(_mixed_seed(seed, salt, combination)).shuffle(inputs)
    targets = tuple(sorted(combination, key=lambda entity: kb[entity]))
    return SortingExample(combination, tuple(inputs), targets)


def validate_dataset_config(config: DatasetConfig) -> None:
    if config.n_entities < 2 or not math.isfinite(config.id_fraction) or not 0 < config.id_fraction < 1:
        raise ValueError("require n_entities >= 2 and id_fraction in (0, 1)")
    if config.value_max < config.value_min or config.set_size < 1:
        raise ValueError("invalid value range or k")
    id_count = round(config.n_entities * config.id_fraction)
    if id_count < config.set_size or config.n_entities - id_count < config.set_size:
        raise ValueError("ID and OOD groups must each contain at least k entities")
    if (config.phi is None) == (config.sorting_train_count is None):
        raise ValueError("set exactly one of phi and sorting_train_count")
    if config.phi is not None and (not math.isfinite(config.phi) or config.phi <= 0):
        raise ValueError("phi must be finite and positive")
    if config.sorting_train_count is not None and config.sorting_train_count < 1:
        raise ValueError("sorting_train_count must be positive")
    if config.validation_count < 1 or config.test_count < 1:
        raise ValueError("validation_count and test_count must be positive")
    if min(config.combination_enumerate_limit, config.max_sampling_attempts,
           config.max_materialized_examples) < 1:
        raise ValueError("dataset safety limits must be positive")


def create_experiment_data(config: DatasetConfig) -> ExperimentData:
    validate_dataset_config(config)
    kb = generate_knowledge_base(config)
    id_entities, ood_entities = split_entities(config)
    sorting_train_count = (
        config.sorting_train_count if config.sorting_train_count is not None
        else round(len(id_entities) * config.phi)  # type: ignore[arg-type]
    )
    if sorting_train_count < 1:
        raise ValueError("sorting training split must be nonempty")
    materialized = config.n_entities + sorting_train_count + config.validation_count + config.test_count
    if materialized > config.max_materialized_examples:
        raise ValueError(
            f"requested {materialized:,} examples exceeds max_materialized_examples="
            f"{config.max_materialized_examples:,}"
        )
    id_combinations = _sample_valid_combinations(
        id_entities, kb, config.set_size, sorting_train_count + config.validation_count,
        _mixed_seed(config.data_seed, _ID_COMBINATION_XOR), set(),
        config.combination_enumerate_limit, config.max_sampling_attempts,
    )
    ood_combinations = _sample_valid_combinations(
        ood_entities, kb, config.set_size, config.test_count,
        _mixed_seed(config.data_seed, _OOD_COMBINATION_XOR), set(),
        config.combination_enumerate_limit, config.max_sampling_attempts,
    )
    atomic_train = tuple(AtomicExample(entity, kb[entity]) for entity in range(config.n_entities))
    id_set, ood_set = set(id_entities), set(ood_entities)
    data = ExperimentData(
        kb, id_entities, ood_entities, atomic_train,
        tuple(x for x in atomic_train if x.entity in id_set),
        tuple(x for x in atomic_train if x.entity in ood_set),
        tuple(_make_sorting_example(c, kb, config.data_seed, _TRAIN_INPUT_XOR)
              for c in id_combinations[:sorting_train_count]),
        tuple(_make_sorting_example(c, kb, config.data_seed, _VALIDATION_INPUT_XOR)
              for c in id_combinations[sorting_train_count:]),
        tuple(_make_sorting_example(c, kb, config.data_seed, _TEST_INPUT_XOR)
              for c in ood_combinations),
        sorting_train_count / len(id_entities),
    )
    vocab = TaskVocabulary(config.n_entities, config.value_min, config.value_max)
    assert_dataset_integrity(data, config, vocab)
    return data


# 4. Serialization, padding, and integrity -------------------------------------


@dataclass(frozen=True)
class SerializedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    is_sorting: bool


@dataclass(frozen=True)
class EncodedDataset:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    is_sorting: torch.Tensor

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def to(self, device: torch.device) -> "EncodedDataset":
        return EncodedDataset(*(getattr(self, field).to(device) for field in (
            "input_ids", "labels", "attention_mask", "is_sorting"
        )))


def _output_only_labels(input_ids: Sequence[int], sep_id: int) -> tuple[int, ...]:
    labels = list(input_ids)
    sep_position = labels.index(sep_id)
    labels[:sep_position + 1] = [IGNORE_INDEX] * (sep_position + 1)
    return tuple(labels)


def serialize_atomic(example: AtomicExample, vocab: TaskVocabulary) -> SerializedExample:
    input_ids = (
        vocab.bos_id, vocab.atom_id, vocab.attr_id, vocab.entity_id(example.entity),
        vocab.sep_id, vocab.value_id(example.value), vocab.eos_id,
    )
    return SerializedExample(input_ids, _output_only_labels(input_ids, vocab.sep_id), False)


def serialize_sorting(example: SortingExample, vocab: TaskVocabulary) -> SerializedExample:
    input_tokens = tuple(vocab.entity_id(entity) for entity in example.inputs)
    if any(vocab.is_value_id(token) for token in input_tokens):
        raise ValueError("sorting input contains a value token")
    input_ids = (
        vocab.bos_id, vocab.sort_id, vocab.attr_id, *input_tokens, vocab.sep_id,
        *(vocab.entity_id(entity) for entity in example.targets), vocab.eos_id,
    )
    return SerializedExample(input_ids, _output_only_labels(input_ids, vocab.sep_id), True)


def assert_dataset_integrity(
    data: ExperimentData, config: DatasetConfig, vocab: TaskVocabulary
) -> None:
    universe = set(range(config.n_entities))
    id_set, ood_set = set(data.id_entities), set(data.ood_entities)
    if id_set & ood_set or id_set | ood_set != universe:
        raise ValueError("ID/OOD entity partition is invalid")
    atomic_entities = [example.entity for example in data.atomic_train]
    if len(atomic_entities) != config.n_entities or set(atomic_entities) != universe:
        raise ValueError("atomic training must contain every entity exactly once")
    atomic_map = {example.entity: example.value for example in data.atomic_train}
    if any(atomic_map.get(entity) != data.kb[entity] for entity in ood_set):
        raise ValueError("an OOD atomic fact is missing or incorrect")
    train_combinations = {x.combination for x in data.sorting_train}
    validation_combinations = {x.combination for x in data.id_validation}
    if len(train_combinations) != len(data.sorting_train) or len(validation_combinations) != len(data.id_validation):
        raise ValueError("duplicate ID sorting combination")
    if train_combinations & validation_combinations:
        raise ValueError("sorting train and ID validation combinations overlap")
    if any(set(x.inputs) - id_set for x in data.sorting_train):
        raise ValueError("sorting training contains a non-ID entity")
    if any(set(x.inputs) - id_set for x in data.id_validation):
        raise ValueError("ID validation contains a non-ID entity")
    if any(set(x.inputs) - ood_set for x in data.ood_test):
        raise ValueError("OOD test contains a non-OOD entity")
    if any(ood_set & set(x.inputs) for x in data.sorting_train):
        raise ValueError("OOD entity leaked into sorting training")
    for name, examples in (
        ("sorting train", data.sorting_train),
        ("ID validation", data.id_validation),
        ("OOD test", data.ood_test),
    ):
        seen: set[tuple[int, ...]] = set()
        for example in examples:
            if example.combination in seen:
                raise ValueError(f"{name} contains a duplicate combination")
            seen.add(example.combination)
            if len({data.kb[e] for e in example.inputs}) != config.set_size:
                raise ValueError(f"{name} example has duplicate values")
            if sorted(example.inputs) != list(example.combination):
                raise ValueError(f"{name} input differs from its combination")
            if sorted(example.targets) != sorted(example.inputs):
                raise ValueError(f"{name} target is not an input permutation")
            if example.targets != tuple(sorted(example.inputs, key=lambda e: data.kb[e])):
                raise ValueError(f"{name} target is not KB-value ascending")
            serialized = serialize_sorting(example, vocab)
            if any(vocab.is_value_id(token) for token in serialized.input_ids[3:3 + config.set_size]):
                raise ValueError(f"{name} sorting input contains a value token")


def encode_examples(
    examples: Sequence[TaskExample], vocab: TaskVocabulary, block_size: int
) -> EncodedDataset:
    if not examples:
        raise ValueError("cannot encode an empty dataset")
    serialized = [
        serialize_sorting(x, vocab) if isinstance(x, SortingExample) else serialize_atomic(x, vocab)
        for x in examples
    ]
    if any(len(x.input_ids) > block_size for x in serialized):
        raise ValueError("serialized example exceeds block_size")
    count = len(serialized)
    input_ids = torch.full((count, block_size), vocab.pad_id, dtype=torch.long)
    labels = torch.full((count, block_size), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((count, block_size), dtype=torch.bool)
    for row, example in enumerate(serialized):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids)
        labels[row, :length] = torch.tensor(example.labels)
        attention_mask[row, :length] = True
    return EncodedDataset(
        input_ids, labels, attention_mask,
        torch.tensor([x.is_sorting for x in serialized], dtype=torch.bool),
    )


def concatenate_encoded(datasets: Sequence[EncodedDataset]) -> EncodedDataset:
    return EncodedDataset(*(torch.cat([getattr(x, field) for x in datasets]) for field in (
        "input_ids", "labels", "attention_mask", "is_sorting"
    )))


def index_encoded(dataset: EncodedDataset, indices: torch.Tensor) -> EncodedDataset:
    return EncodedDataset(*(getattr(dataset, field)[indices] for field in (
        "input_ids", "labels", "attention_mask", "is_sorting"
    )))


def dataset_fingerprint(data: ExperimentData) -> str:
    payload = {
        "kb": data.kb, "id": data.id_entities, "ood": data.ood_entities,
        "train": [asdict(x) for x in data.sorting_train],
        "validation": [asdict(x) for x in data.id_validation],
        "test": [asdict(x) for x in data.ood_test],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


# 5. Shared GPT-2-style decoder ------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    set_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    bias: bool = True
    tie_embeddings: bool = False
    init_std: float = 0.02
    layer_norm_epsilon: float = 1e-5
    architecture: str = "causal-gpt2-entity-kb-sort-v1"

    @property
    def block_size(self) -> int:
        return 2 * self.set_size + 5


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.n_embd, config.n_head, dropout=config.dropout,
            bias=config.bias, batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor,
                key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        output, _ = self.attention(
            x, x, x, attn_mask=causal_mask,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        return self.dropout(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(F.gelu(self.fc(x), approximate="tanh")))


class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor,
                key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), causal_mask, key_padding_mask)
        return x + self.mlp(self.mlp_norm(x))


class GPTSortTransformer(nn.Module):
    """One unconstrained causal Transformer shared by both tasks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size < 1 or config.n_embd < 1 or config.n_head < 1 or config.n_embd % config.n_head:
            raise ValueError("invalid vocabulary/embedding/head dimensions")
        if config.n_layer < 1:
            raise ValueError("n_layer must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.block_size:
            raise ValueError("input_ids shape exceeds model block_size")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        causal_mask = torch.triu(torch.ones(
            input_ids.shape[1], input_ids.shape[1], dtype=torch.bool,
            device=input_ids.device,
        ), diagonal=1)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)
        return self.lm_head(self.final_norm(x))

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 eos_id: int, pad_id: int) -> torch.Tensor:
        if prompt_ids.ndim != 2 or max_new_tokens < 1:
            raise ValueError("invalid generation request")
        if prompt_ids.shape[1] + max_new_tokens > self.config.block_size:
            raise ValueError("generation exceeds model block_size")
        context = prompt_ids
        generated: list[torch.Tensor] = []
        finished = torch.zeros(len(prompt_ids), dtype=torch.bool, device=prompt_ids.device)
        for _ in range(max_new_tokens):
            next_token = self(context)[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, pad_id, next_token)
            generated.append(next_token)
            finished |= next_token.eq(eos_id)
            context = torch.cat((context, next_token[:, None]), dim=1)
            if bool(finished.all()):
                generated.extend(torch.full_like(next_token, pad_id)
                                 for _ in range(max_new_tokens - len(generated)))
                break
        return torch.stack(generated, dim=1)


EntityKBTransformer = GPTSortTransformer
SortTransformer = GPTSortTransformer


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                   label_smoothing: float = 0.0, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing, reduction=reduction,
    )


# 6. Evaluation ----------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    loss: float
    token_acc: float
    exact_acc: float
    invalid_token_rate: float
    early_eos_rate: float
    duplicate_rate: float
    not_in_input_rate: float


def batches(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


@torch.inference_mode()
def evaluate_encoded(
    model: GPTSortTransformer, dataset: EncodedDataset, task: str,
    vocab: TaskVocabulary, set_size: int, batch_size: int,
    device: torch.device, dtype_name: str,
) -> Metrics:
    if task not in ("atomic", "sorting") or len(dataset) < 1:
        raise ValueError("invalid evaluation task or empty dataset")
    model.eval()
    loss_sum = 0.0
    token_correct = token_count = exact_correct = invalid_count = 0
    early_eos_count = duplicate_count = not_in_input_count = 0
    generated_count = entity_slot_count = 0
    for selection in batches(len(dataset), batch_size):
        input_ids = dataset.input_ids[selection].to(device, non_blocking=device.type == "cuda")
        labels = dataset.labels[selection].to(device, non_blocking=device.type == "cuda")
        attention_mask = dataset.attention_mask[selection].to(
            device, non_blocking=device.type == "cuda"
        )
        with amp_context(device, dtype_name):
            logits = model(input_ids, attention_mask)
            shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
            supervised = shifted_labels.ne(IGNORE_INDEX)
            loss_sum += causal_lm_loss(logits, labels, reduction="sum").item()
        predictions = shifted_logits.argmax(dim=-1)
        token_correct += (predictions.eq(shifted_labels) & supervised).sum().item()
        token_count += supervised.sum().item()
        prompt_length = 5 if task == "atomic" else set_size + 4
        output_length = 2 if task == "atomic" else set_size + 1
        prompt = input_ids[:, :prompt_length]
        expected = input_ids[:, prompt_length:prompt_length + output_length]
        generated = model.generate(prompt, output_length, vocab.eos_id, vocab.pad_id)
        exact_correct += generated.eq(expected).all(dim=1).sum().item()
        generated_count += generated.numel()
        if task == "atomic":
            valid = torch.stack((
                generated[:, 0].ge(vocab.value_offset) & generated[:, 0].lt(vocab.bos_id),
                generated[:, 1].eq(vocab.eos_id),
            ), dim=1)
            early_eos_count += generated[:, 0].eq(vocab.eos_id).sum().item()
        else:
            entity_outputs = generated[:, :set_size]
            entity_valid = entity_outputs.ge(0) & entity_outputs.lt(vocab.n_entities)
            valid = torch.cat((entity_valid, generated[:, -1:].eq(vocab.eos_id)), dim=1)
            early_eos_count += entity_outputs.eq(vocab.eos_id).any(dim=1).sum().item()
            duplicate_pairs = entity_outputs[:, :, None].eq(
                entity_outputs[:, None, :]
            ) & entity_valid[:, :, None] & entity_valid[:, None, :]
            duplicate_pairs = torch.triu(duplicate_pairs, diagonal=1)
            duplicate_count += duplicate_pairs.flatten(1).any(dim=1).sum().item()
            source = input_ids[:, 3:3 + set_size]
            in_input = entity_outputs.unsqueeze(-1).eq(source.unsqueeze(1)).any(dim=-1)
            not_in_input_count += (entity_valid & ~in_input).sum().item()
            entity_slot_count += entity_outputs.numel()
        invalid_count += (~valid).sum().item()
    return Metrics(
        loss_sum / token_count, token_correct / token_count,
        exact_correct / len(dataset), invalid_count / generated_count,
        early_eos_count / len(dataset),
        duplicate_count / len(dataset) if task == "sorting" else 0.0,
        not_in_input_count / entity_slot_count if task == "sorting" else 0.0,
    )


def combine_metrics(weighted: Sequence[tuple[Metrics, int]]) -> Metrics:
    total = sum(count for _, count in weighted)
    if total < 1:
        raise ValueError("cannot combine empty metrics")
    return Metrics(*(sum(getattr(metric, field) * count for metric, count in weighted) / total
                     for field in Metrics.__dataclass_fields__))


def subset_encoded(dataset: EncodedDataset, count: int) -> EncodedDataset:
    if count < 0 or count >= len(dataset):
        return dataset
    return index_encoded(dataset, torch.arange(count))


# 7. Fixed task mixture and optimization ---------------------------------------


def fixed_batch_indices(length: int, batch_size: int, step: int,
                        device: torch.device) -> torch.Tensor:
    if length < 1 or step < 1 or batch_size == 0 or batch_size < -1:
        raise ValueError("invalid fixed-batch settings")
    if batch_size == -1:
        return torch.arange(length, device=device)
    start = ((step - 1) * batch_size) % length
    return torch.arange(start, start + batch_size, device=device) % length


def _controlled_batch_sizes(batch_size: int, atomic_fraction: float) -> tuple[int, int]:
    atomic_count = round(batch_size * atomic_fraction)
    sorting_count = batch_size - atomic_count
    if atomic_count < 1 or sorting_count < 1:
        raise ValueError("controlled batch must contain both atomic and sorting examples")
    return atomic_count, sorting_count


def build_training_batch(
    step: int, args: argparse.Namespace, combined: EncodedDataset,
    atomic: EncodedDataset, sorting: EncodedDataset,
    controlled_order: torch.Tensor | None,
) -> EncodedDataset:
    if args.training_mode == "combined":
        indices = fixed_batch_indices(
            len(combined), args.batch_size, step, combined.input_ids.device
        )
        return index_encoded(combined, indices)
    assert controlled_order is not None
    atomic_count, sorting_count = _controlled_batch_sizes(
        args.batch_size, args.atomic_fraction
    )
    atomic_indices = fixed_batch_indices(
        len(atomic), atomic_count, step, atomic.input_ids.device
    )
    sorting_indices = fixed_batch_indices(
        len(sorting), sorting_count, step, sorting.input_ids.device
    )
    batch = concatenate_encoded((
        index_encoded(atomic, atomic_indices), index_encoded(sorting, sorting_indices)
    ))
    return index_encoded(batch, controlled_order)


def dynamically_permute_sorting_inputs(
    batch: EncodedDataset, set_size: int, generator: torch.Generator
) -> EncodedDataset:
    rows = batch.is_sorting.nonzero(as_tuple=False).flatten()
    if len(rows) == 0:
        return batch
    input_ids = batch.input_ids.clone()
    values = input_ids[rows, 3:3 + set_size]
    order = torch.rand(values.shape, device=values.device, generator=generator).argsort(dim=1)
    input_ids[rows, 3:3 + set_size] = values.gather(1, order)
    return EncodedDataset(input_ids, batch.labels, batch.attention_mask, batch.is_sorting)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if dtype_name == "bfloat16" else torch.float16)


def parameter_norm(model: nn.Module) -> float:
    squares = sum(parameter.detach().float().square().sum() for parameter in model.parameters())
    return math.sqrt(squares.item())


def lr_for_step(args: argparse.Namespace, step: int) -> float:
    warmup_scale = min(1.0, step / max(1, args.warmup))
    progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    if args.lr_schedule == "cosine":
        decay = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    elif args.lr_schedule == "linear":
        decay = 1.0 - min(1.0, progress)
    else:
        decay = 1.0
    return args.lr * warmup_scale * decay


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    common = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), betas=(args.beta1, args.beta2), **common)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), betas=(args.beta1, args.beta2), **common)
    return torch.optim.SGD(model.parameters(), momentum=args.momentum, **common)


# 8. Logging/checkpoints --------------------------------------------------------


def signature_digest(signature: dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def validate_log_target(path: Path, resume_step: int | None, signature: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"log CSV schema does not match: {path}")
        rows = list(reader)
    if resume_step is None:
        raise ValueError(f"log CSV already contains a run: {path}")
    steps = [int(row["step"]) for row in rows]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError(f"log CSV has non-increasing steps: {path}")
    if steps and steps[-1] > resume_step:
        raise ValueError(f"log CSV continues past resume step {resume_step}: {path}")
    if any(row["run_signature_sha256"] != signature for row in rows):
        raise ValueError(f"log CSV belongs to another run: {path}")


def save_checkpoint(
    path: Path, model: GPTSortTransformer, optimizer: torch.optim.Optimizer,
    model_config: ModelConfig, dataset_config: DatasetConfig,
    run_signature: dict[str, object], scaler: object | None, step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2, "step": step,
        "model_config": asdict(model_config), "dataset_config": asdict(dataset_config),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "run_signature": run_signature,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# 9. Training ------------------------------------------------------------------


def _metric_columns(prefix: str, metric: Metrics) -> dict[str, float]:
    return {
        f"{prefix}_loss": metric.loss,
        f"{prefix}_token_acc": metric.token_acc,
        f"{prefix}_exact_acc": metric.exact_acc,
    }


def _diagnostic_columns(prefix: str, metric: Metrics) -> dict[str, float]:
    return {
        f"{prefix}_invalid_token_rate": metric.invalid_token_rate,
        f"{prefix}_early_eos_rate": metric.early_eos_rate,
        f"{prefix}_duplicate_rate": metric.duplicate_rate,
        f"{prefix}_not_in_input_rate": metric.not_in_input_rate,
    }


def print_dataset_summary(data: ExperimentData, config: DatasetConfig,
                          vocab: TaskVocabulary, preview: int) -> None:
    print(f"Entities: {config.n_entities:,}")
    print(f"ID entities: {len(data.id_entities):,}")
    print(f"OOD entities: {len(data.ood_entities):,}")
    print("Attributes: 1")
    print(f"Values: {config.value_min}..{config.value_max}")
    print(f"Atomic train examples: {len(data.atomic_train):,}")
    print(f"ID sorting train examples: {len(data.sorting_train):,}")
    print(f"ID validation sorting examples: {len(data.id_validation):,}")
    print(f"OOD sorting test examples: {len(data.ood_test):,}")
    print(f"k: {config.set_size}")
    print(f"Atomic OOD entities seen in atomic training: {len(data.atomic_ood):,}/{len(data.ood_entities):,}")
    leaked = len(set(data.ood_entities) & {e for x in data.sorting_train for e in x.inputs})
    print(f"OOD entities seen in sorting training: {leaked:,}/{len(data.ood_entities):,}")
    print("Sorting train tuple source: ID only")
    print("Sorting test tuple source: OOD only")
    print(f"phi effective: {data.phi_effective:.6g}")
    if preview < 1:
        return
    print("--- dataset preview ---")
    for example in data.atomic_ood[:preview]:
        prompt = [vocab.bos_id, vocab.atom_id, vocab.attr_id, example.entity, vocab.sep_id]
        target = [vocab.value_id(example.value), vocab.eos_id]
        print(f"ATOMIC OOD: {vocab.decode(prompt)} -> {vocab.decode(target)}")
    for label, examples in (("SORT TRAIN (ID)", data.sorting_train),
                            ("SORT TEST (OOD)", data.ood_test)):
        for example in examples[:preview]:
            values = [data.kb[e] for e in example.inputs]
            print(f"{label}: {vocab.decode(example.inputs)} values={values} target={vocab.decode(example.targets)}")


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    dataset_config = DatasetConfig(
        args.n_entities, args.id_fraction, args.value_min, args.value_max,
        args.set_size, args.phi, args.sorting_train_count,
        args.validation_count, args.test_count, args.data_seed,
        args.combination_enumerate_limit, args.max_sampling_attempts,
        args.max_materialized_examples,
    )
    vocab = TaskVocabulary(args.n_entities, args.value_min, args.value_max)
    data = create_experiment_data(dataset_config)
    print_dataset_summary(data, dataset_config, vocab, args.preview)
    model_config = ModelConfig(
        vocab.size, args.set_size, args.n_embd, args.n_head, args.n_layer,
        args.dropout, not args.no_bias, args.tie, args.init_std,
        args.layer_norm_epsilon,
    )
    atomic_all = encode_examples(data.atomic_train, vocab, model_config.block_size)
    atomic_id = encode_examples(data.atomic_id, vocab, model_config.block_size)
    atomic_ood = encode_examples(data.atomic_ood, vocab, model_config.block_size)
    sorting_train = encode_examples(data.sorting_train, vocab, model_config.block_size)
    id_validation = encode_examples(data.id_validation, vocab, model_config.block_size)
    ood_test = encode_examples(data.ood_test, vocab, model_config.block_size)
    combined = concatenate_encoded((atomic_all, sorting_train))
    generator = torch.Generator().manual_seed(args.seed ^ 0x6A09E667)
    combined = index_encoded(combined, torch.randperm(len(combined), generator=generator))
    controlled_order = None
    if args.training_mode == "controlled":
        atomic_batch, _ = _controlled_batch_sizes(args.batch_size, args.atomic_fraction)
        generator = torch.Generator().manual_seed(args.seed ^ 0xBB67AE85)
        controlled_order = torch.randperm(args.batch_size, generator=generator)
        effective_atomic_fraction = atomic_batch / args.batch_size
    else:
        effective_atomic_fraction = len(atomic_all) / len(combined)

    model = GPTSortTransformer(model_config).to(device)
    optimizer = build_optimizer(model, args)
    run_signature: dict[str, object] = {
        "architecture": model_config.architecture,
        "model_config": asdict(model_config),
        "dataset_sha256": dataset_fingerprint(data),
        "optimizer": args.optimizer, "seed": args.seed,
        "optimizer_config": {
            "lr": args.lr, "beta1": args.beta1, "beta2": args.beta2,
            "momentum": args.momentum, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "warmup": args.warmup,
            "lr_schedule": args.lr_schedule,
            "label_smoothing": args.label_smoothing,
        },
        "dtype": args.dtype,
        "training_mode": args.training_mode,
        "atomic_fraction": (
            args.atomic_fraction if args.training_mode == "controlled" else None
        ),
        "effective_atomic_fraction": effective_atomic_fraction,
        "dynamic_input_permutation": args.dynamic_input_permutation,
        "batch_size": args.batch_size, "data_order": "fixed-v1",
    }
    signature = signature_digest(run_signature)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and args.dtype == "float16" else None
    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != asdict(model_config):
            raise ValueError("checkpoint model configuration does not match")
        if checkpoint.get("dataset_config") != asdict(dataset_config):
            raise ValueError("checkpoint dataset configuration does not match")
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError("checkpoint run signature does not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if args.steps <= start_step:
            raise ValueError("steps must exceed checkpoint step")
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    combined_gpu = combined.to(device)
    atomic_gpu, sorting_gpu = atomic_all.to(device), sorting_train.to(device)
    if controlled_order is not None:
        controlled_order = controlled_order.to(device)
    executable: nn.Module = model
    if args.compile and device.type == "cuda":
        executable = torch.compile(model, mode=args.compile_mode, dynamic=False)
    eval_train = subset_encoded(sorting_train, args.n_eval)
    eval_validation = subset_encoded(id_validation, args.n_eval)
    eval_test = subset_encoded(ood_test, args.n_eval)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        special_names = ("BOS", "SEP", "EOS", "PAD", "ATOM", "SORT", "ATTR")
        payload = {
            "model": asdict(model_config), "dataset": asdict(dataset_config),
            "training": vars(args), "vocabulary_size": vocab.size,
            "special_tokens": {name: getattr(vocab, f"{name.lower()}_id") for name in special_names},
            "knowledge_base": list(data.kb), "id_entities": list(data.id_entities),
            "ood_entities": list(data.ood_entities),
            "dataset_sha256": run_signature["dataset_sha256"],
        }
        (out_dir / "config.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    if args.log_csv:
        validate_log_target(Path(args.log_csv), start_step if args.resume else None, signature)

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "-"
    print(f"device={device} gpu={gpu} dtype={args.dtype if device.type == 'cuda' else 'float32'} compile={args.compile and device.type == 'cuda'}")
    print(f"architecture={model_config.architecture} vocab={vocab.size:,} params={sum(p.numel() for p in model.parameters()):,}")
    print(f"training_mode={args.training_mode} effective_atomic_fraction={effective_atomic_fraction:.6f} data_order=fixed dynamic_input_permutation={args.dynamic_input_permutation}")

    training_size = len(combined)
    effective_batch_size = training_size if args.batch_size == -1 else args.batch_size
    examples_seen = start_step * effective_batch_size
    dynamic_generator = torch.Generator(device=device)
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        learning_rate = lr_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = build_training_batch(
            step, args, combined_gpu, atomic_gpu, sorting_gpu, controlled_order
        )
        if args.dynamic_input_permutation:
            dynamic_generator.manual_seed(args.seed + step)
            batch = dynamically_permute_sorting_inputs(batch, args.set_size, dynamic_generator)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.dtype):
            logits = executable(batch.input_ids, batch.attention_mask)
            loss = causal_lm_loss(logits, batch.labels, args.label_smoothing)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        examples_seen += len(batch)

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            atomic_id_m = evaluate_encoded(model, atomic_id, "atomic", vocab, args.set_size, args.eval_batch, device, args.dtype)
            atomic_ood_m = evaluate_encoded(model, atomic_ood, "atomic", vocab, args.set_size, args.eval_batch, device, args.dtype)
            atomic_train_m = combine_metrics(((atomic_id_m, len(atomic_id)), (atomic_ood_m, len(atomic_ood))))
            sorting_train_m = evaluate_encoded(model, eval_train, "sorting", vocab, args.set_size, args.eval_batch, device, args.dtype)
            validation_m = evaluate_encoded(model, eval_validation, "sorting", vocab, args.set_size, args.eval_batch, device, args.dtype)
            test_m = evaluate_encoded(model, eval_test, "sorting", vocab, args.set_size, args.eval_batch, device, args.dtype)
            total_loss = effective_atomic_fraction * atomic_train_m.loss + (1 - effective_atomic_fraction) * sorting_train_m.loss
            epoch, norm = examples_seen / training_size, parameter_norm(model)
            elapsed = time.perf_counter() - started
            print(f"step {step:7,d}/{args.steps:,d} epoch {epoch:,.2f} lr {learning_rate:.2e} |w| {norm:.2f} | loss total {total_loss:.4f} atom {atomic_train_m.loss:.4f} sort {sorting_train_m.loss:.4f}")
            print(f"  atomic exact ID {atomic_id_m.exact_acc:.3f} OOD {atomic_ood_m.exact_acc:.3f} | sorting exact train {sorting_train_m.exact_acc:.3f} ID-val {validation_m.exact_acc:.3f} OOD-test {test_m.exact_acc:.3f} | {elapsed:.1f}s")
            if args.log_csv:
                row: dict[str, object] = {
                    "step": step, "epoch": epoch, "examples_seen": examples_seen,
                    "lr": learning_rate, "weight_norm": norm,
                    "train_loss": total_loss, "train_token_acc": sorting_train_m.token_acc,
                    "train_exact_acc": sorting_train_m.exact_acc,
                    "validation_loss": validation_m.loss,
                    "validation_token_acc": validation_m.token_acc,
                    "validation_exact_acc": validation_m.exact_acc,
                    "test_loss": test_m.loss, "test_token_acc": test_m.token_acc,
                    "test_exact_acc": test_m.exact_acc,
                    **_metric_columns("atomic_train", atomic_train_m),
                    **_metric_columns("atomic_id", atomic_id_m),
                    **_metric_columns("atomic_ood", atomic_ood_m),
                    **_metric_columns("sorting_train", sorting_train_m),
                    **_metric_columns("id_validation_sorting", validation_m),
                    **_metric_columns("ood_test_sorting", test_m),
                    "atomic_id_invalid_token_rate": atomic_id_m.invalid_token_rate,
                    "atomic_id_early_eos_rate": atomic_id_m.early_eos_rate,
                    "atomic_ood_invalid_token_rate": atomic_ood_m.invalid_token_rate,
                    "atomic_ood_early_eos_rate": atomic_ood_m.early_eos_rate,
                    **_diagnostic_columns("sorting_train", sorting_train_m),
                    **_diagnostic_columns("id_validation", validation_m),
                    **_diagnostic_columns("ood_test", test_m),
                    "training_mode": args.training_mode,
                    "configured_atomic_fraction": args.atomic_fraction if args.training_mode == "controlled" else "",
                    "effective_atomic_fraction": effective_atomic_fraction,
                    "dynamic_input_permutation": args.dynamic_input_permutation,
                    "elapsed_seconds": elapsed,
                    "train_eval_count": len(eval_train),
                    "validation_eval_count": len(eval_validation),
                    "test_eval_count": len(eval_test),
                    "atomic_id_eval_count": len(atomic_id),
                    "atomic_ood_eval_count": len(atomic_ood),
                    "run_signature_sha256": signature,
                }
                write_csv(Path(args.log_csv), row)
        if out_dir and args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(out_dir / f"ckpt_{step:08d}.pt", model, optimizer,
                            model_config, dataset_config, run_signature, scaler, step)

    if out_dir:
        save_checkpoint(out_dir / "ckpt_final.pt", model, optimizer,
                        model_config, dataset_config, run_signature, scaler, args.steps)
    sample_count = min(5, len(ood_test))
    sample = index_encoded(ood_test, torch.arange(sample_count))
    prompt_length = args.set_size + 4
    generated = model.generate(
        sample.input_ids[:, :prompt_length].to(device), args.set_size + 1,
        vocab.eos_id, vocab.pad_id,
    ).cpu()
    print("--- OOD sorting samples ---")
    for row in range(sample_count):
        inputs = sample.input_ids[row, 3:3 + args.set_size].tolist()
        expected = sample.input_ids[row, prompt_length:prompt_length + args.set_size + 1].tolist()
        actual = generated[row].tolist()
        print(f"in {vocab.decode(inputs)} -> {vocab.decode(actual)} expected {vocab.decode(expected)} {'OK' if actual == expected else 'MISS'}")
    return 0


# 10. CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--n-entities", type=int, default=1000)
    parser.add_argument("--id-fraction", type=float, default=0.9)
    parser.add_argument("--value-min", type=int, default=0)
    parser.add_argument("--value-max", type=int, default=100)
    parser.add_argument("--k", "--m", dest="set_size", type=int, default=3)
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument("--phi", type=float)
    sizing.add_argument("--sorting-train-count", type=int)
    parser.add_argument("--validation-count", "--val-count", dest="validation_count", type=int, default=2000)
    parser.add_argument("--test-count", "--n-test", dest="test_count", type=int, default=5000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--combination-enumerate-limit", type=int, default=5_000_000)
    parser.add_argument("--max-sampling-attempts", type=int, default=2_000_000)
    parser.add_argument("--max-materialized-examples", type=int, default=500_000)
    parser.add_argument("--training-mode", choices=("combined", "controlled"), default="combined")
    parser.add_argument("--atomic-fraction", type=float, default=0.25)
    parser.add_argument("--dynamic-input-permutation", action="store_true")
    parser.add_argument("--n-embd", "--hidden-dim", dest="n_embd", type=int, default=128)
    parser.add_argument("--n-head", "--attention-heads", dest="n_head", type=int, default=4)
    parser.add_argument("--n-layer", "--layers", dest="n_layer", type=int, default=2)
    parser.add_argument("--n-enc-layer", type=int, default=0, help="compatibility option; must be 0")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    parser.add_argument("--output-constraint", choices=("free",), default="free")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine", "linear"), default="constant")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--preview", type=int, default=2)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.phi is None and args.sorting_train_count is None:
        args.phi = 3.6
    if args.n_embd < 1 or args.n_head < 1 or args.n_embd % args.n_head:
        raise ValueError("n-embd must be positive and divisible by n-head")
    if args.n_layer < 1 or args.n_enc_layer != 0:
        raise ValueError("n-layer must be positive and n-enc-layer must be 0")
    if not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1:
        raise ValueError("dropout and label-smoothing must be in [0, 1)")
    if args.steps < 1 or args.batch_size == 0 or args.batch_size < -1:
        raise ValueError("steps must be positive and batch-size must be -1 or positive")
    if args.training_mode == "controlled":
        if args.batch_size == -1:
            raise ValueError("controlled training requires positive batch-size")
        if not math.isfinite(args.atomic_fraction) or not 0 < args.atomic_fraction < 1:
            raise ValueError("atomic-fraction must be in (0, 1)")
        _controlled_batch_sizes(args.batch_size, args.atomic_fraction)
    if args.eval_every < 1 or args.eval_batch < 1 or args.n_eval == 0 or args.n_eval < -1:
        raise ValueError("invalid evaluation settings")
    if min(args.preview, args.warmup, args.ckpt_every) < 0:
        raise ValueError("preview, warmup, and ckpt-every must be nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("invalid optimizer settings")
    if not 0 <= args.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    validate_dataset_config(DatasetConfig(
        args.n_entities, args.id_fraction, args.value_min, args.value_max,
        args.set_size, args.phi, args.sorting_train_count,
        args.validation_count, args.test_count, args.data_seed,
        args.combination_enumerate_limit, args.max_sampling_attempts,
        args.max_materialized_examples,
    ))


def apply_smoke_settings(args: argparse.Namespace) -> None:
    args.n_entities, args.id_fraction = 20, 0.8
    args.value_min, args.value_max, args.set_size = 0, 10, 3
    args.phi, args.sorting_train_count = None, 20
    args.validation_count, args.test_count = 10, 1
    args.n_embd, args.n_head, args.n_layer = 32, 4, 1
    args.training_mode, args.atomic_fraction = "combined", 0.25
    args.dynamic_input_permutation = False
    args.steps, args.batch_size = 20, -1
    args.eval_every, args.n_eval, args.eval_batch = 10, -1, 128
    args.warmup, args.device, args.compile, args.preview = 2, "cpu", False, 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        apply_smoke_settings(args)
    try:
        validate_args(args)
        return train(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
