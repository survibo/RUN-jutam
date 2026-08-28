"""Sort unordered fixed-size sets with positional cross-attention queries.

Edit only CONFIG, then run ``python microgpt_set.py``. Each non-empty data
line is one set, for example ``{1, 3, 5, 7, 11}``.
"""

from __future__ import annotations

import itertools
import math
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    data: Path
    min_token: int
    max_token: int
    set_size: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    steps: int
    learning_rate: float
    weight_decay: float
    report_every: int
    training_batch_size: int
    validation_batch_size: int
    data_build_chunk_size: int
    use_pinned_memory: bool
    use_amp: bool
    amp_dtype: str
    compile_model: bool
    compile_mode: str
    seed: int
    device: str
    predict: tuple[tuple[int, ...], ...]

    @property
    def vocabulary_size(self) -> int:
        return self.max_token - self.min_token + 1

    @property
    def all_tokens(self) -> tuple[int, ...]:
        return tuple(range(self.min_token, self.max_token + 1))


# -----------------------------------------------------------------------------
# 모든 실행 설정: 이 블록의 값만 수정하면 됩니다.
# -----------------------------------------------------------------------------
CONFIG = Config(
    data=Path("example.txt"),          # 학습 데이터 파일
    min_token=1,                       # 전체 집합의 최솟값
    max_token=50,                      # 전체 집합의 최댓값 (현재 1~50)
    set_size=6,                        # 입력 집합 및 출력 수열 길이
    steps=2_000,                       # 전체 full-batch optimizer 스텝
    report_every=100,                  # 전수 train/validation 로그 간격
    learning_rate=3e-4,                # 학습률
    weight_decay=1e-2,                 # AdamW weight decay
    d_model=128,                       # 토큰 임베딩 및 모델 차원
    n_heads=4,                         # cross-attention head 개수
    n_layers=2,                        # cross-attention + MLP 블록 개수
    mlp_ratio=4,                       # MLP 내부 차원 배율
    training_batch_size=32_768,        # full-batch gradient 누적 GPU 청크
    validation_batch_size=131_072,     # 전수 validation GPU 청크
    data_build_chunk_size=262_144,     # CPU 데이터 사전 생성 청크
    use_pinned_memory=True,            # CPU→GPU 비동기 전송용 고정 메모리
    use_amp=True,                      # CUDA mixed precision
    amp_dtype="float16",              # "float16" 또는 "bfloat16"
    compile_model=True,                # torch.compile 사용 여부
    compile_mode="max-autotune",       # torch.compile 모드
    seed=42,                           # 랜덤 시드
    device="auto",                    # "auto", "cuda", "cpu"
    predict=((7, 2, 6, 4, 5, 9),),    # 학습 후 예측. 없으면 ()
)


def validate_config(config: Config) -> None:
    if config.max_token < config.min_token:
        raise ValueError("max_token must be greater than or equal to min_token")
    if not 1 <= config.set_size <= config.vocabulary_size:
        raise ValueError("set_size must be between 1 and vocabulary size")
    if config.d_model <= 0 or config.n_heads <= 0:
        raise ValueError("d_model and n_heads must be positive")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if config.n_layers <= 0 or config.mlp_ratio <= 0:
        raise ValueError("n_layers and mlp_ratio must be positive")
    if config.steps <= 0 or config.report_every <= 0:
        raise ValueError("steps and report_every must be positive")
    if min(
        config.training_batch_size,
        config.validation_batch_size,
        config.data_build_chunk_size,
    ) <= 0:
        raise ValueError("training/validation/build batch sizes must be positive")
    if config.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError('amp_dtype must be "float16" or "bfloat16"')
    if config.compile_mode not in {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }:
        raise ValueError("unsupported compile_mode")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError('device must be "auto", "cpu", or "cuda"')


def parse_set(text: str) -> tuple[int, ...]:
    values = tuple(int(piece) for piece in re.findall(r"[-+]?\d+", text))
    if not values:
        raise ValueError(f"no integer token found in {text!r}")
    return values


def canonicalize(
    values: Sequence[int], config: Config, source: str
) -> tuple[int, ...]:
    if len(values) != config.set_size:
        raise ValueError(
            f"{source}: expected {config.set_size} tokens, found {len(values)}"
        )
    if len(set(values)) != config.set_size:
        raise ValueError(f"{source}: repeated tokens are not allowed")
    outside = [
        value
        for value in values
        if value < config.min_token or value > config.max_token
    ]
    if outside:
        raise ValueError(
            f"{source}: tokens {outside} are outside "
            f"[{config.min_token}, {config.max_token}]"
        )
    return tuple(sorted(values))


def compact_integer_dtype(vocabulary_size: int) -> torch.dtype:
    if vocabulary_size <= torch.iinfo(torch.int16).max + 1:
        return torch.int16
    if vocabulary_size <= torch.iinfo(torch.int32).max:
        return torch.int32
    return torch.int64


def allocate_host_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, use_pinned_memory: bool
) -> torch.Tensor:
    if use_pinned_memory:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError as error:
            print(f"warning: pinned allocation failed ({error}); using normal RAM")
    return torch.empty(shape, dtype=dtype)


def combination_ranks(token_ids: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Return zero-based lexicographic ranks for sorted combinations."""
    choose = torch.zeros((n + 1, k + 1), dtype=torch.int64)
    for population in range(n + 1):
        for selection in range(min(population, k) + 1):
            choose[population, selection] = math.comb(population, selection)

    ids = token_ids.to(torch.int64)
    previous = torch.cat(
        (torch.full((len(ids), 1), -1, dtype=torch.int64), ids[:, :-1]), dim=1
    )
    ranks = torch.zeros(len(ids), dtype=torch.int64)
    for position in range(k):
        order = k - position
        ranks += (
            choose[n - previous[:, position] - 1, order]
            - choose[n - ids[:, position], order]
        )
    return ranks


@dataclass(frozen=True)
class TrainingData:
    targets: torch.Tensor
    membership: bytearray


def load_training_data(
    config: Config, total_combinations: int, use_pinned_memory: bool
) -> TrainingData:
    if not config.data.is_file():
        raise FileNotFoundError(f"training data not found: {config.data}")
    with config.data.open("r", encoding="utf-8") as data_file:
        training_size = sum(1 for line in data_file if line.strip())
    if training_size == 0:
        raise ValueError(f"no training sets found in {config.data}")
    if training_size > total_combinations:
        raise ValueError("training rows exceed the number of possible combinations")

    dtype = compact_integer_dtype(config.vocabulary_size)
    targets = allocate_host_tensor(
        (training_size, config.set_size), dtype, use_pinned_memory
    )
    membership = bytearray(total_combinations)
    buffer: list[tuple[int, ...]] = []
    written = 0
    next_progress = 10
    started_at = time.perf_counter()

    def flush_buffer() -> None:
        nonlocal buffer, written, next_progress
        if not buffer:
            return
        chunk = torch.tensor(buffer, dtype=dtype)
        chunk.sub_(config.min_token)
        ranks = combination_ranks(
            chunk, config.vocabulary_size, config.set_size
        ).tolist()
        for offset, rank in enumerate(ranks):
            if membership[rank]:
                raise ValueError(
                    f"duplicate training set near row {written + offset + 1}"
                )
            membership[rank] = 1
        end = written + len(buffer)
        targets[written:end].copy_(chunk)
        written = end
        buffer = []
        progress = written * 100 // training_size
        if progress >= next_progress or written == training_size:
            print(
                f"loading training: {written:,}/{training_size:,} ({progress}%)",
                end="\r" if written < training_size else "\n",
                flush=True,
            )
            next_progress = (progress // 10 + 1) * 10

    with config.data.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            line = line.strip()
            if not line:
                continue
            buffer.append(
                canonicalize(
                    parse_set(line), config, f"{config.data}:{line_number}"
                )
            )
            if len(buffer) == config.data_build_chunk_size:
                flush_buffer()
    flush_buffer()

    elapsed = time.perf_counter() - started_at
    memory_mib = targets.numel() * targets.element_size() / (1024**2)
    print(
        f"training prepared in {elapsed:.1f}s | CPU memory={memory_mib:.1f} MiB "
        f"| pinned={targets.is_pinned()}"
    )
    return TrainingData(targets=targets, membership=membership)


def build_validation_targets(
    config: Config,
    training_membership: bytearray,
    validation_size: int,
    use_pinned_memory: bool,
) -> torch.Tensor:
    """Build every non-training combination once for all exhaustive checks."""
    dtype = compact_integer_dtype(config.vocabulary_size)
    targets = allocate_host_tensor(
        (validation_size, config.set_size), dtype, use_pinned_memory
    )
    buffer: list[tuple[int, ...]] = []
    written = 0
    next_progress = 10
    started_at = time.perf_counter()

    def flush_buffer() -> None:
        nonlocal buffer, written, next_progress
        if not buffer:
            return
        chunk = torch.tensor(buffer, dtype=dtype)
        chunk.sub_(config.min_token)
        end = written + len(buffer)
        targets[written:end].copy_(chunk)
        written = end
        buffer = []
        progress = written * 100 // validation_size
        if progress >= next_progress or written == validation_size:
            print(
                f"building validation: {written:,}/{validation_size:,} ({progress}%)",
                end="\r" if written < validation_size else "\n",
                flush=True,
            )
            next_progress = (progress // 10 + 1) * 10

    for rank, candidate in enumerate(
        itertools.combinations(config.all_tokens, config.set_size)
    ):
        if training_membership[rank]:
            continue
        buffer.append(candidate)
        if len(buffer) == config.data_build_chunk_size:
            flush_buffer()
    flush_buffer()
    if written != validation_size:
        raise RuntimeError(
            f"validation size mismatch: expected {validation_size}, got {written}"
        )

    elapsed = time.perf_counter() - started_at
    memory_mib = targets.numel() * targets.element_size() / (1024**2)
    print(
        f"validation prepared once in {elapsed:.1f}s | CPU memory={memory_mib:.1f} "
        f"MiB | pinned={targets.is_pinned()}"
    )
    return targets


class CrossAttentionMLPBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attention_output, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + attention_output
        return queries + self.mlp(self.mlp_norm(queries))


class SetSorter(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.output_position_queries = nn.Parameter(
            torch.empty(config.set_size, config.d_model)
        )
        self.blocks = nn.ModuleList(
            CrossAttentionMLPBlock(
                config.d_model, config.n_heads, config.mlp_ratio
            )
            for _ in range(config.n_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.vocabulary_head = nn.Linear(config.d_model, config.vocabulary_size)
        nn.init.normal_(self.output_position_queries, mean=0.0, std=0.02)

    def forward(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        # No input position embedding: input order is irrelevant.
        memory = self.token_embedding(input_token_ids)
        queries = self.output_position_queries.unsqueeze(0).expand(
            input_token_ids.shape[0], -1, -1
        )
        for block in self.blocks:
            queries = block(queries, memory)
        # No pointer mask: each position predicts from the full vocabulary.
        return self.vocabulary_head(self.output_norm(queries))


def to_token_ids(
    sets: Sequence[Sequence[int]], config: Config, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [[token - config.min_token for token in item] for item in sets],
        dtype=torch.long,
        device=device,
    )


def shuffled_rows(values: torch.Tensor) -> torch.Tensor:
    permutation = torch.rand(values.shape, device=values.device).argsort(dim=1)
    return values.gather(dim=1, index=permutation)


@dataclass(frozen=True)
class Metrics:
    loss: float
    token_correct: int
    sequence_correct: int
    sequence_count: int
    token_count: int

    @property
    def mean_loss(self) -> float:
        return self.loss / self.token_count

    @property
    def token_accuracy(self) -> float:
        return self.token_correct / self.token_count

    @property
    def exact_accuracy(self) -> float:
        return self.sequence_correct / self.sequence_count


class DeviceMetricAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.loss = torch.zeros((), dtype=torch.float64, device=device)
        self.token_correct = torch.zeros((), dtype=torch.int64, device=device)
        self.sequence_correct = torch.zeros((), dtype=torch.int64, device=device)
        self.sequence_count = 0
        self.token_count = 0

    def add(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        loss = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum"
        )
        matches = logits.argmax(dim=-1).eq(targets)
        self.loss.add_(loss.to(torch.float64))
        self.token_correct.add_(matches.sum())
        self.sequence_correct.add_(matches.all(dim=1).sum())
        self.sequence_count += targets.shape[0]
        self.token_count += targets.numel()

    def finalize(self) -> Metrics:
        values = torch.stack(
            (
                self.loss,
                self.token_correct.to(torch.float64),
                self.sequence_correct.to(torch.float64),
            )
        ).cpu().tolist()
        return Metrics(
            loss=float(values[0]),
            token_correct=int(values[1]),
            sequence_correct=int(values[2]),
            sequence_count=self.sequence_count,
            token_count=self.token_count,
        )


def amp_context(config: Config, device: torch.device) -> ContextManager[object]:
    if not config.use_amp or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def device_batches(
    host_targets: torch.Tensor, batch_size: int, device: torch.device
) -> Iterator[torch.Tensor]:
    """Prefetch compact pinned batches while the current batch is computed."""
    if device.type != "cuda" or not host_targets.is_pinned():
        for start in range(0, len(host_targets), batch_size):
            yield host_targets[start : start + batch_size].to(
                device=device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )
        return

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    def copy_to_device(start: int) -> torch.Tensor:
        with torch.cuda.stream(copy_stream):
            compact = host_targets[start : start + batch_size].to(
                device=device, non_blocking=True
            )
            return compact.to(dtype=torch.long)

    next_start = 0
    next_batch: torch.Tensor | None = copy_to_device(next_start)
    next_start += batch_size
    while next_batch is not None:
        compute_stream.wait_stream(copy_stream)
        current_batch = next_batch
        next_batch = (
            copy_to_device(next_start) if next_start < len(host_targets) else None
        )
        next_start += batch_size
        current_batch.record_stream(compute_stream)
        yield current_batch


@torch.inference_mode()
def evaluate_dataset(
    model: nn.Module,
    host_targets: torch.Tensor,
    batch_size: int,
    config: Config,
    device: torch.device,
) -> Metrics:
    model.eval()
    accumulator = DeviceMetricAccumulator(device)
    for targets in device_batches(host_targets, batch_size, device):
        with amp_context(config, device):
            accumulator.add(model(targets), targets)
    return accumulator.finalize()


def print_report(
    step: int, train: Metrics, validation: Metrics, validation_seconds: float
) -> None:
    print(
        f"step {step:6d} | "
        f"train loss {train.mean_loss:.6f} | "
        f"train token acc {train.token_accuracy:.4%} | "
        f"train exact acc {train.exact_accuracy:.4%} | "
        f"val loss {validation.mean_loss:.6f} | "
        f"val token acc {validation.token_accuracy:.4%} | "
        f"val exact acc {validation.exact_accuracy:.4%} | "
        f"val time {validation_seconds:.1f}s"
    )


@torch.inference_mode()
def predict(
    model: nn.Module,
    values: Sequence[int],
    config: Config,
    device: torch.device,
) -> None:
    original = tuple(values)
    canonicalize(original, config, f"prediction {original!r}")
    input_ids = to_token_ids([original], config, device)
    with amp_context(config, device):
        output_ids = model(input_ids).argmax(dim=-1).squeeze(0).tolist()
    output_tokens = tuple(token_id + config.min_token for token_id in output_ids)
    print(f"input {set(original)} -> output {output_tokens}")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError('device="cuda" was configured, but CUDA is unavailable')
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grad_scaler(config: Config, device: torch.device):
    enabled = config.use_amp and config.amp_dtype == "float16"
    if not enabled or device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def main() -> None:
    config = CONFIG
    validate_config(config)
    for index, values in enumerate(config.predict):
        canonicalize(values, config, f"CONFIG.predict[{index}]")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")

    device = choose_device(config.device)
    use_pinned = config.use_pinned_memory and device.type == "cuda"
    total_combinations = math.comb(config.vocabulary_size, config.set_size)
    training_data = load_training_data(config, total_combinations, use_pinned)
    validation_size = total_combinations - len(training_data.targets)
    if validation_size <= 0:
        raise ValueError("validation is empty: training contains every possible set")
    validation_targets = build_validation_targets(
        config, training_data.membership, validation_size, use_pinned
    )

    print(
        f"device={device} | vocabulary={config.min_token}..{config.max_token} "
        f"| set_size={config.set_size}"
    )
    print(
        f"train={len(training_data.targets):,} (full-batch via accumulation) | "
        f"validation={len(validation_targets):,} (exhaustive, train excluded)"
    )

    model = SetSorter(config).to(device)
    compiled = False
    if config.compile_model and device.type == "cuda":
        if hasattr(torch, "compile"):
            print(f"torch.compile mode={config.compile_mode!r} (first calls are slow)")
            model = torch.compile(model, mode=config.compile_mode, dynamic=False)
            compiled = True
        else:
            print("warning: torch.compile unavailable; using eager mode")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = make_grad_scaler(config, device)
    amp_enabled = config.use_amp and device.type == "cuda"
    print(
        f"AMP={'on (' + config.amp_dtype + ')' if amp_enabled else 'off'} | "
        f"compile={'on' if compiled else 'off'} | "
        f"train chunk={config.training_batch_size:,} | "
        f"validation chunk={config.validation_batch_size:,}"
    )

    total_training_tokens = training_data.targets.numel()
    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # All chunks contribute to one update. Global normalization makes this
        # mathematically the same gradient as a single full-batch forward pass.
        for targets in device_batches(
            training_data.targets, config.training_batch_size, device
        ):
            with amp_context(config, device):
                logits = model(shuffled_rows(targets))
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                    reduction="sum",
                ) / total_training_tokens
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        if step % config.report_every == 0 or step == config.steps:
            train_metrics = evaluate_dataset(
                model,
                training_data.targets,
                config.validation_batch_size,
                config,
                device,
            )
            validation_started_at = time.perf_counter()
            validation_metrics = evaluate_dataset(
                model,
                validation_targets,
                config.validation_batch_size,
                config,
                device,
            )
            validation_seconds = time.perf_counter() - validation_started_at
            print_report(
                step, train_metrics, validation_metrics, validation_seconds
            )

    model.eval()
    for values in config.predict:
        predict(model, values, config, device)


if __name__ == "__main__":
    main()
