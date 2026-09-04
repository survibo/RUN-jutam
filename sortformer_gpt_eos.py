"""Train a GPT-2-style causal Transformer to autoregressively sort numbers.

The serialized language-model sequence is::

    [BOS, input_1, ..., input_k, SEP, sorted_1, ..., sorted_k, EOS]

Every number is one discrete token.  Labels through SEP are masked with
``IGNORE_INDEX``, so loss is computed only for the sorted suffix and EOS.
Train, validation, and test splits contain disjoint unordered combinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import sortdata


IGNORE_INDEX = -100
CSV_COLUMNS = (
    "step",
    "epoch",
    "lr",
    "weight_norm",
    "train_loss",
    "train_token_acc",
    "train_exact_acc",
    "validation_loss",
    "validation_token_acc",
    "validation_exact_acc",
    "test_loss",
    "test_token_acc",
    "test_exact_acc",
    "elapsed_seconds",
    "train_eval_count",
    "validation_eval_count",
    "test_eval_count",
    "run_signature_sha256",
)


# ---------------------------------------------------------------------------
# 1. Vocabulary / tokenizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberVocabulary:
    """A one-token-per-number vocabulary plus explicit special tokens."""

    numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.numbers:
            raise ValueError("the number vocabulary must not be empty")
        if len(set(self.numbers)) != len(self.numbers):
            raise ValueError("number vocabulary entries must be unique")
        if tuple(sorted(self.numbers)) != self.numbers:
            raise ValueError("numbers must be strictly increasing")

    @classmethod
    def contiguous(cls, size: int, minimum: int = 0) -> "NumberVocabulary":
        if size < 1:
            raise ValueError("vocabulary size must be positive")
        return cls(tuple(range(minimum, minimum + size)))

    @property
    def number_token_count(self) -> int:
        return len(self.numbers)

    @property
    def bos_id(self) -> int:
        return len(self.numbers)

    @property
    def sep_id(self) -> int:
        return len(self.numbers) + 1

    @property
    def eos_id(self) -> int:
        return len(self.numbers) + 2

    @property
    def size(self) -> int:
        return len(self.numbers) + 3

    def encode_number(self, value: int) -> int:
        try:
            return self.numbers.index(value)
        except ValueError as exc:
            raise ValueError(f"number is not in the vocabulary: {value}") from exc

    def decode_id(self, token_id: int) -> int | str:
        if 0 <= token_id < len(self.numbers):
            return self.numbers[token_id]
        specials = {
            self.bos_id: "BOS",
            self.sep_id: "SEP",
            self.eos_id: "EOS",
        }
        if token_id not in specials:
            raise ValueError(f"token id is outside the vocabulary: {token_id}")
        return specials[token_id]

    def decode(self, token_ids: Sequence[int]) -> list[int | str]:
        return [self.decode_id(int(token_id)) for token_id in token_ids]


# ---------------------------------------------------------------------------
# 2-3. Dataset generation and disjoint train/validation/test split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    n: int = 20
    set_size: int = 4
    minimum: int = 0
    train_count: int | None = 128
    train_percent: float | None = None
    validation_count: int = -1
    test_count: int = -1
    split_strategy: str = "random"
    seed: int = 0
    max_materialized_examples: int = 500_000


@dataclass(frozen=True)
class SortingExample:
    """One input permutation and its ascending target, both as token IDs."""

    combination_rank: int
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[SortingExample, ...]
    validation: tuple[SortingExample, ...]
    test: tuple[SortingExample, ...]
    total_combinations: int


_SPLIT_KEYS = {
    "train": 0xA0761D6478BD642F,
    "validation": 0xE7037ED1A0B428DB,
    "test": 0x8EBC6AF09C88C6E3,
}


def _sample_unique(total: int, count: int, rng: random.Random) -> list[int]:
    """Floyd sampling without enumerating a potentially huge combination space."""
    if count < 0 or count > total:
        raise ValueError("sample size is outside the available combinations")
    selected: set[int] = set()
    result: list[int] = []
    for upper in range(total - count, total):
        candidate = rng.randrange(upper + 1)
        chosen = upper if candidate in selected else candidate
        selected.add(chosen)
        result.append(chosen)
    rng.shuffle(result)
    return result


def _resolve_split_counts(config: DatasetConfig, total: int) -> tuple[int, int, int]:
    if (config.train_count is None) == (config.train_percent is None):
        raise ValueError("set exactly one of train_count and train_percent")
    if config.train_count is not None:
        train_count = config.train_count
    else:
        assert config.train_percent is not None
        if not math.isfinite(config.train_percent) or not 0 <= config.train_percent <= 100:
            raise ValueError("train_percent must be finite and in [0, 100]")
        train_count = round(total * config.train_percent / 100)
    if train_count < 1 or train_count >= total:
        raise ValueError("training split must be nonempty and leave held-out examples")

    remaining = total - train_count
    validation_count, test_count = config.validation_count, config.test_count
    if validation_count < -1 or test_count < -1:
        raise ValueError("validation_count and test_count must be -1 or nonnegative")
    if validation_count == -1 and test_count == -1:
        validation_count = remaining // 2
        test_count = remaining - validation_count
    elif validation_count == -1:
        validation_count = remaining - test_count
    elif test_count == -1:
        test_count = remaining - validation_count
    if validation_count < 1 or test_count < 1:
        raise ValueError("validation and test splits must both be nonempty")
    if train_count + validation_count + test_count > total:
        raise ValueError("requested split sizes exceed the combination space")
    materialized = train_count + validation_count + test_count
    if materialized > config.max_materialized_examples:
        raise ValueError(
            f"requested {materialized:,} examples exceeds max_materialized_examples="
            f"{config.max_materialized_examples:,}; set explicit held-out counts"
        )
    return train_count, validation_count, test_count


def _permutation_seed(seed: int, rank: int, split: str) -> int:
    value = (seed ^ _SPLIT_KEYS[split] ^ (rank * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _make_example(rank: int, config: DatasetConfig, split: str) -> SortingExample:
    targets = sortdata.combination_unrank(rank, config.n, config.set_size)
    inputs = list(targets)
    random.Random(_permutation_seed(config.seed, rank, split)).shuffle(inputs)
    return SortingExample(rank, tuple(inputs), targets)


def create_dataset_splits(config: DatasetConfig) -> DatasetSplits:
    """Create deterministic splits disjoint by underlying unordered combination."""
    if config.n < 1 or config.set_size < 1 or config.set_size > config.n:
        raise ValueError("require n >= 1 and 1 <= set_size <= n")
    if config.max_materialized_examples < 3:
        raise ValueError("max_materialized_examples must be at least 3")
    if config.split_strategy not in ("random", "lexicographic"):
        raise ValueError("split_strategy must be random or lexicographic")
    total = math.comb(config.n, config.set_size)
    counts = _resolve_split_counts(config, total)
    requested_count = sum(counts)
    if config.split_strategy == "random":
        ranks = _sample_unique(total, requested_count, random.Random(config.seed))
    else:
        ranks = list(range(requested_count))
    train_end = counts[0]
    validation_end = train_end + counts[1]
    rank_splits = {
        "train": ranks[:train_end],
        "validation": ranks[train_end:validation_end],
        "test": ranks[validation_end:],
    }
    examples = {
        split: tuple(_make_example(rank, config, split) for rank in split_ranks)
        for split, split_ranks in rank_splits.items()
    }
    return DatasetSplits(
        examples["train"], examples["validation"], examples["test"], total
    )


def examples_to_tensors(
    examples: Sequence[SortingExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([example.inputs for example in examples], dtype=torch.long),
        torch.tensor([example.targets for example in examples], dtype=torch.long),
    )


def serialize_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocabulary: NumberVocabulary,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full LM sequences and labels masked through the separator."""
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise ValueError("inputs and targets must be equal-shaped rank-2 tensors")
    batch_size, set_size = inputs.shape
    bos = torch.full((batch_size, 1), vocabulary.bos_id, device=inputs.device, dtype=torch.long)
    sep = torch.full((batch_size, 1), vocabulary.sep_id, device=inputs.device, dtype=torch.long)
    eos = torch.full((batch_size, 1), vocabulary.eos_id, device=inputs.device, dtype=torch.long)
    input_ids = torch.cat((bos, inputs, sep, targets, eos), dim=1)
    labels = input_ids.clone()
    # The shifted causal loss first becomes active when SEP predicts target[0].
    labels[:, : set_size + 2] = IGNORE_INDEX
    return input_ids, labels


def shuffled_rows(
    inputs: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    order = torch.rand(inputs.shape, device=inputs.device, generator=generator).argsort(dim=1)
    return inputs.gather(1, order)


# ---------------------------------------------------------------------------
# 4. GPT-2-style decoder-only Transformer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Model dimensions; vocab_size counts number tokens, not special tokens."""

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
    architecture: str = "causal-gpt2-sort-eos-v1"

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size + 3

    @property
    def block_size(self) -> int:
        return 2 * self.set_size + 3


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(
            x, x, x, attn_mask=causal_mask, need_weights=False
        )
        return self.residual_dropout(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc(x), approximate="tanh")
        return self.dropout(self.projection(x))


class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            config.n_embd, eps=config.layer_norm_epsilon
        )
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), causal_mask)
        return x + self.mlp(self.mlp_norm(x))


class GPTSortTransformer(nn.Module):
    """A single causal self-attention stack with no sorting-specific modules."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size < config.set_size:
            raise ValueError("vocab_size must be at least set_size")
        if config.n_embd < 1 or config.n_head < 1 or config.n_embd % config.n_head:
            raise ValueError("n_embd must be positive and divisible by n_head")
        if config.n_layer < 1:
            raise ValueError("n_layer must be positive")
        self.config = config
        self.bos_id = config.vocab_size
        self.sep_id = config.vocab_size + 1
        self.eos_id = config.vocab_size + 2
        self.token_embedding = nn.Embedding(config.total_vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.total_vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 tensor")
        if input_ids.shape[1] < 1 or input_ids.shape[1] > self.config.block_size:
            raise ValueError(
                f"sequence length must be in [1, {self.config.block_size}]"
            )
        if torch.compiler.is_compiling() is False and (
            input_ids.min().item() < 0
            or input_ids.max().item() >= self.config.total_vocab_size
        ):
            raise ValueError("input_ids contain a token outside the vocabulary")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        causal_mask = torch.triu(
            torch.ones(
                input_ids.shape[1],
                input_ids.shape[1],
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.lm_head(self.final_norm(x))

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Greedily generate exactly ``set_size + 1`` slots, ending with EOS."""
        if inputs.ndim != 2 or inputs.shape[1] != self.config.set_size:
            raise ValueError(f"inputs must have shape [batch, {self.config.set_size}]")
        bos = torch.full(
            (inputs.shape[0], 1), self.bos_id, dtype=torch.long, device=inputs.device
        )
        sep = torch.full(
            (inputs.shape[0], 1), self.sep_id, dtype=torch.long, device=inputs.device
        )
        context = torch.cat((bos, inputs, sep), dim=1)
        generated: list[torch.Tensor] = []
        finished = torch.zeros(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        for _ in range(self.config.set_size + 1):
            next_token = self(context)[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, self.eos_id, next_token)
            generated.append(next_token)
            finished |= next_token.eq(self.eos_id)
            context = torch.cat((context, next_token[:, None]), dim=1)
            if bool(finished.all()):
                remaining = self.config.set_size + 1 - len(generated)
                generated.extend(
                    torch.full_like(next_token, self.eos_id) for _ in range(remaining)
                )
                break
        return torch.stack(generated, dim=1)


# Alias retained for the naming convention used by sortformer.py.
SortTransformer = GPTSortTransformer


def causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Shift logits/labels and apply the output-only label mask."""
    if logits.shape[:2] != labels.shape or logits.ndim != 3:
        raise ValueError("logits and labels have incompatible shapes")
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )


# ---------------------------------------------------------------------------
# 5-8. Training, generation, evaluation, logging, and checkpointing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    loss: float
    token_acc: float
    exact_acc: float


def batches(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def parameter_norm(model: nn.Module) -> float:
    squares = sum(
        parameter.detach().float().square().sum() for parameter in model.parameters()
    )
    return math.sqrt(squares.item())


@torch.inference_mode()
def evaluate(
    model: GPTSortTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocabulary: NumberVocabulary,
    batch_size: int,
    device: torch.device,
    dtype_name: str,
) -> Metrics:
    """Measure teacher-forced token accuracy and free-running exact accuracy."""
    if len(inputs) == 0:
        raise ValueError("cannot evaluate an empty split")
    model.eval()
    loss_sum = 0.0
    token_correct = 0
    token_count = 0
    exact_correct = 0
    for selection in batches(len(inputs), batch_size):
        x = inputs[selection].to(device, non_blocking=device.type == "cuda")
        y = targets[selection].to(device, non_blocking=device.type == "cuda")
        input_ids, labels = serialize_batch(x, y, vocabulary)
        with amp_context(device, dtype_name):
            logits = model(input_ids)
            shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
            supervised = shifted_labels.ne(IGNORE_INDEX)
            loss_sum += F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            ).item()
        predictions = shifted_logits.argmax(dim=-1)
        token_correct += (predictions.eq(shifted_labels) & supervised).sum().item()
        token_count += supervised.sum().item()
        generated = model.generate(x)
        expected = torch.cat(
            (
                y,
                torch.full(
                    (len(y), 1), vocabulary.eos_id, dtype=torch.long, device=device
                ),
            ),
            dim=1,
        )
        exact_correct += generated.eq(expected).all(dim=1).sum().item()
    return Metrics(
        loss=loss_sum / token_count,
        token_acc=token_correct / token_count,
        exact_acc=exact_correct / len(inputs),
    )


def subset_for_eval(
    inputs: torch.Tensor, targets: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 0 or count >= len(inputs):
        return inputs, targets
    return inputs[:count], targets[:count]


def dataset_fingerprint(splits: DatasetSplits) -> str:
    digest = hashlib.sha256()
    for name, examples in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        digest.update(name.encode("ascii"))
        for example in examples:
            digest.update(f"{example.combination_rank}\n".encode("ascii"))
    return digest.hexdigest()


def signature_digest(signature: dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def validate_log_target(
    path: Path, resume_step: int | None, run_signature_sha256: str
) -> None:
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
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"log CSV has non-increasing steps: {path}")
    if steps and steps[-1] > resume_step:
        raise ValueError(f"log CSV continues past resume step {resume_step}: {path}")
    if any(row["run_signature_sha256"] != run_signature_sha256 for row in rows):
        raise ValueError(f"log CSV belongs to a different run: {path}")


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


def build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    common = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), betas=(args.beta1, args.beta2), **common
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), betas=(args.beta1, args.beta2), **common
        )
    return torch.optim.SGD(model.parameters(), momentum=args.momentum, **common)


def save_checkpoint(
    path: Path,
    model: GPTSortTransformer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    run_signature: dict[str, object],
    scaler: object | None,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "step": step,
        "model_config": asdict(model_config),
        "dataset_config": asdict(dataset_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "run_signature": run_signature,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    vocabulary = NumberVocabulary.contiguous(args.n, args.minimum)
    dataset_config = DatasetConfig(
        n=args.n,
        set_size=args.set_size,
        minimum=args.minimum,
        train_count=args.train_count,
        train_percent=args.train_percent,
        validation_count=args.validation_count,
        test_count=args.test_count,
        split_strategy=args.split_strategy,
        seed=args.data_seed,
        max_materialized_examples=args.max_materialized_examples,
    )
    splits = create_dataset_splits(dataset_config)
    train_x, train_y = examples_to_tensors(splits.train)
    validation_x, validation_y = examples_to_tensors(splits.validation)
    test_x, test_y = examples_to_tensors(splits.test)
    if device.type == "cuda":
        train_x, train_y = train_x.to(device), train_y.to(device)
        validation_x, validation_y = validation_x.pin_memory(), validation_y.pin_memory()
        test_x, test_y = test_x.pin_memory(), test_y.pin_memory()

    model_config = ModelConfig(
        vocab_size=args.n,
        set_size=args.set_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        bias=not args.no_bias,
        tie_embeddings=args.tie,
        init_std=args.init_std,
        layer_norm_epsilon=args.layer_norm_epsilon,
    )
    model = GPTSortTransformer(model_config).to(device)
    optimizer = build_optimizer(model, args)
    run_signature: dict[str, object] = {
        "architecture": model_config.architecture,
        "model_config": asdict(model_config),
        "dataset_sha256": dataset_fingerprint(splits),
        "number_vocabulary": list(vocabulary.numbers),
        "optimizer": args.optimizer,
        "seed": args.seed,
    }
    run_signature_sha256 = signature_digest(run_signature)

    scaler = None
    if device.type == "cuda" and args.dtype == "float16":
        scaler = torch.amp.GradScaler("cuda")
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
            raise ValueError("steps must be greater than the resumed checkpoint step")
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    executable: nn.Module = model
    if args.compile and device.type == "cuda":
        executable = torch.compile(model, mode=args.compile_mode, dynamic=False)
    effective_batch_size = len(train_x) if args.batch_size == -1 else args.batch_size
    eval_train_x, eval_train_y = subset_for_eval(train_x, train_y, args.n_eval)
    eval_validation_x, eval_validation_y = subset_for_eval(
        validation_x, validation_y, args.n_eval
    )
    eval_test_x, eval_test_y = subset_for_eval(test_x, test_y, args.n_eval)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": asdict(model_config),
                    "dataset": asdict(dataset_config),
                    "training": vars(args),
                    "special_tokens": {
                        "BOS": vocabulary.bos_id,
                        "SEP": vocabulary.sep_id,
                        "EOS": vocabulary.eos_id,
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    if args.log_csv:
        validate_log_target(
            Path(args.log_csv), start_step if args.resume else None, run_signature_sha256
        )

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "-"
    print(
        f"device={device} gpu={gpu} dtype={args.dtype if device.type == 'cuda' else 'float32'} "
        f"compile={args.compile and device.type == 'cuda'}"
    )
    print(
        f"task=ascending architecture={model_config.architecture} "
        f"train={len(train_x):,} validation={len(validation_x):,} test={len(test_x):,} "
        f"C({args.n},{args.set_size})={splits.total_combinations:,} | "
        f"params={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    print(
        f"sequence=[BOS,{args.set_size} inputs,SEP,{args.set_size} targets,EOS] "
        f"supervised_tokens/example={args.set_size + 1}"
    )

    full_batch_indices = (
        torch.arange(len(train_x), device=train_x.device)
        if args.batch_size == -1
        else None
    )
    step_generator = torch.Generator(device=train_x.device)
    started = time.perf_counter()
    examples_seen = start_step * effective_batch_size
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        learning_rate = lr_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        step_generator.manual_seed(args.seed + step)
        if full_batch_indices is None:
            indices = torch.randint(
                len(train_x),
                (effective_batch_size,),
                device=train_x.device,
                generator=step_generator,
            )
        else:
            indices = full_batch_indices
        x = shuffled_rows(train_x[indices], step_generator)
        y = train_y[indices]
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        input_ids, labels = serialize_batch(x, y, vocabulary)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.dtype):
            logits = executable(input_ids)
            loss = causal_lm_loss(logits, labels, args.label_smoothing)
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
        examples_seen += len(x)

        should_eval = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_eval:
            train_metrics = evaluate(
                model,
                eval_train_x,
                eval_train_y,
                vocabulary,
                args.eval_batch,
                device,
                args.dtype,
            )
            validation_metrics = evaluate(
                model,
                eval_validation_x,
                eval_validation_y,
                vocabulary,
                args.eval_batch,
                device,
                args.dtype,
            )
            test_metrics = evaluate(
                model,
                eval_test_x,
                eval_test_y,
                vocabulary,
                args.eval_batch,
                device,
                args.dtype,
            )
            epoch = examples_seen / len(train_x)
            norm = parameter_norm(model)
            elapsed = time.perf_counter() - started
            print(
                f"step {step:7,d}/{args.steps:,d} epoch {epoch:,.2f} "
                f"lr {learning_rate:.2e} |w| {norm:.2f} | "
                f"train loss {train_metrics.loss:.4f} tok {train_metrics.token_acc:.3f} "
                f"exact {train_metrics.exact_acc:.3f} | "
                f"val loss {validation_metrics.loss:.4f} tok {validation_metrics.token_acc:.3f} "
                f"exact {validation_metrics.exact_acc:.3f} | "
                f"test loss {test_metrics.loss:.4f} tok {test_metrics.token_acc:.3f} "
                f"exact {test_metrics.exact_acc:.3f} | {elapsed:.1f}s"
            )
            if args.log_csv:
                write_csv(
                    Path(args.log_csv),
                    {
                        "step": step,
                        "epoch": epoch,
                        "lr": learning_rate,
                        "weight_norm": norm,
                        "train_loss": train_metrics.loss,
                        "train_token_acc": train_metrics.token_acc,
                        "train_exact_acc": train_metrics.exact_acc,
                        "validation_loss": validation_metrics.loss,
                        "validation_token_acc": validation_metrics.token_acc,
                        "validation_exact_acc": validation_metrics.exact_acc,
                        "test_loss": test_metrics.loss,
                        "test_token_acc": test_metrics.token_acc,
                        "test_exact_acc": test_metrics.exact_acc,
                        "elapsed_seconds": elapsed,
                        "train_eval_count": len(eval_train_x),
                        "validation_eval_count": len(eval_validation_x),
                        "test_eval_count": len(eval_test_x),
                        "run_signature_sha256": run_signature_sha256,
                    },
                )
        if out_dir and args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(
                out_dir / f"ckpt_{step:08d}.pt",
                model,
                optimizer,
                model_config,
                dataset_config,
                run_signature,
                scaler,
                step,
            )

    if out_dir:
        save_checkpoint(
            out_dir / "ckpt_final.pt",
            model,
            optimizer,
            model_config,
            dataset_config,
            run_signature,
            scaler,
            args.steps,
        )
    sample_count = min(5, len(test_x))
    sample_inputs = test_x[:sample_count].to(device)
    generated = model.generate(sample_inputs).cpu()
    print("--- unseen combinations ---")
    for input_row, target_row, generated_row in zip(
        test_x[:sample_count], test_y[:sample_count], generated
    ):
        decoded_input = vocabulary.decode(input_row.tolist())
        expected = vocabulary.decode(target_row.tolist() + [vocabulary.eos_id])
        actual = vocabulary.decode(generated_row.tolist())
        status = "OK" if actual == expected else "MISS"
        print(f"in {decoded_input} -> {actual} expected {expected} {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task", choices=("ascending",), default="ascending")
    parser.add_argument("--n", type=int, default=20, help="number-token vocabulary size")
    parser.add_argument("--minimum", type=int, default=0, help="smallest represented number")
    parser.add_argument("--k", "--m", dest="set_size", type=int, default=4)
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument("--train-count", type=int)
    sizing.add_argument("--train-percent", type=float)
    parser.add_argument("--validation-count", "--val-count", dest="validation_count", type=int, default=-1)
    parser.add_argument("--test-count", "--n-test", dest="test_count", type=int, default=-1)
    parser.add_argument("--split-strategy", choices=("random", "lexicographic"), default="random")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--max-materialized-examples", type=int, default=500_000)
    parser.add_argument("--n-embd", "--hidden-dim", dest="n_embd", type=int, default=128)
    parser.add_argument("--n-head", "--attention-heads", dest="n_head", type=int, default=4)
    parser.add_argument("--n-layer", "--layers", dest="n_layer", type=int, default=2)
    parser.add_argument("--n-enc-layer", type=int, default=0, help="compatibility option; must remain 0")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    parser.add_argument("--output-constraint", choices=("free",), default="free")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="positive: sample with replacement; -1: use every training row per step",
    )
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
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.train_count is None and args.train_percent is None:
        args.train_count = 128
    if args.n < 2 or args.set_size < 1 or args.set_size > args.n:
        raise ValueError("require n >= 2 and 1 <= k <= n")
    if args.n_embd < 1 or args.n_head < 1 or args.n_embd % args.n_head:
        raise ValueError("n-embd must be positive and divisible by n-head")
    if args.n_layer < 1 or args.n_enc_layer != 0:
        raise ValueError("n-layer must be positive and n-enc-layer must be 0")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be in [0, 1)")
    if args.steps < 1 or args.batch_size == 0 or args.batch_size < -1:
        raise ValueError("steps must be positive and batch-size must be -1 or positive")
    if args.eval_every < 1 or args.eval_batch < 1 or args.n_eval == 0 or args.n_eval < -1:
        raise ValueError("evaluation settings must be positive, with n-eval optionally -1")
    if args.warmup < 0 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("optimizer rates and intervals are outside their valid range")
    if not 0 <= args.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if args.ckpt_every < 0:
        raise ValueError("ckpt-every must be nonnegative")


def apply_smoke_settings(args: argparse.Namespace) -> None:
    args.n, args.set_size = 7, 3
    args.train_count, args.train_percent = 18, None
    args.validation_count, args.test_count = 8, 9
    args.n_embd, args.n_head, args.n_layer = 32, 4, 1
    args.steps, args.batch_size = 20, -1
    args.eval_every, args.n_eval, args.eval_batch = 10, -1, 128
    args.warmup, args.device, args.compile = 2, "cpu", False


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
