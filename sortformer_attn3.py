"""Train an autoregressive Transformer on set-to-permutation sorting tasks.

Attention ablation 3/3: decoder cross-attention only. Neither the encoder nor
the decoder has self-attention.
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

from sortdata import (
    DatasetConfig,
    SortExample,
    classify_test_example,
    combination_rank,
    compute_train_comparison_information,
    iter_examples,
    load_dataset,
    split_ranks,
)


TASKS = ("ascending", "mod", "alternating")
OUTPUT_CONSTRAINTS = ("permutation", "input-only", "free")
STRATA = ("direct", "transitive", "unresolved")
CSV_COLUMNS = (
    "step", "lr", "weight_norm", "train_loss", "train_token_acc",
    "train_gen_in_set_token_acc", "train_set_acc", "train_exact_acc",
    "test_loss", "test_token_acc", "test_gen_in_set_token_acc",
    "test_set_acc", "test_exact_acc", "elapsed_seconds", "task",
    "output_constraint", "run_signature_sha256", "train_eval_count",
    "test_eval_count", "test_direct_exact_acc", "test_transitive_exact_acc",
    "test_unresolved_exact_acc", "test_direct_count", "test_transitive_count",
    "test_unresolved_count",
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    set_size: int
    n_embd: int = 128
    n_head: int = 4
    n_enc_layer: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    output_constraint: str = "permutation"
    tie_embeddings: bool = False
    init_std: float = 0.02
    init_scale: float = 1.0


@dataclass(frozen=True)
class Metrics:
    loss: float
    token_acc: float
    gen_in_set_token_acc: float
    set_acc: float
    exact_acc: float
    strata_exact_acc: dict[str, float | None]
    strata_counts: dict[str, int]


class RMSNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + 1e-5
        )
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            size, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.attn(
            query, key_value, key_value, attn_mask=mask, need_weights=False
        )
        return self.dropout(output)


class MLP(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, 4 * size, bias=False),
            nn.ReLU(),
            nn.Linear(4 * size, size, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.mlp_norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cross_norm = RMSNorm(config.n_embd)
        self.memory_norm = RMSNorm(config.n_embd)
        self.cross_attn = Attention(config.n_embd, config.n_head, config.dropout)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.cross_norm(x), self.memory_norm(memory))
        return x + self.mlp(self.mlp_norm(x))


class SortTransformer(nn.Module):
    """Attention-free set encoder; the decoder reads it by cross-attention only."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.bos_id = config.vocab_size
        self.encoder_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.decoder_embedding = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.decoder_position = nn.Embedding(config.set_size, config.n_embd)
        self.encoder = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.n_enc_layer)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.n_layer)
        )
        self.encoder_norm = RMSNorm(config.n_embd)
        self.decoder_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.encoder_embedding.weight
        if config.init_scale != 1.0:
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.mul_(config.init_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        # Deliberately no input position embedding: this is a set encoder.
        memory = self.encoder_embedding(inputs)
        for block in self.encoder:
            memory = block(memory)
        return self.encoder_norm(memory)

    def _valid_token_mask(
        self, inputs: torch.Tensor, decoder_inputs: torch.Tensor
    ) -> torch.Tensor:
        batch, length = decoder_inputs.shape
        membership = torch.zeros(
            batch, self.config.vocab_size, dtype=torch.bool, device=inputs.device
        )
        membership.scatter_(1, inputs, True)
        valid = membership[:, None, :].expand(-1, length, -1)
        if self.config.output_constraint == "input-only":
            return valid
        seen = torch.zeros_like(membership)
        masks = []
        for position in range(length):
            token = decoder_inputs[:, position]
            seen.scatter_(
                1,
                token.clamp_max(self.config.vocab_size - 1).unsqueeze(1),
                token.ne(self.bos_id).unsqueeze(1),
            )
            masks.append(membership & ~seen)
        return torch.stack(masks, dim=1)

    def forward(self, inputs: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        memory = self.encode(inputs)
        positions = torch.arange(decoder_inputs.shape[1], device=inputs.device)
        x = self.decoder_embedding(decoder_inputs) + self.decoder_position(positions)
        for block in self.decoder:
            x = block(x, memory)
        logits = self.lm_head(self.decoder_norm(x))
        if self.config.output_constraint != "free":
            logits = logits.masked_fill(
                ~self._valid_token_mask(inputs, decoder_inputs), float("-inf")
            )
        return logits

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor) -> torch.Tensor:
        decoder_inputs = torch.full(
            (inputs.shape[0], 1), self.bos_id, dtype=torch.long, device=inputs.device
        )
        generated = []
        for _ in range(self.config.set_size):
            next_token = self(inputs, decoder_inputs)[:, -1].argmax(dim=-1)
            generated.append(next_token)
            decoder_inputs = torch.cat((decoder_inputs, next_token[:, None]), dim=1)
        return torch.stack(generated, dim=1)


def targets_for(examples: Sequence[SortExample], task: str) -> torch.Tensor:
    attribute = {"ascending": "asc", "mod": "mod", "alternating": "alt"}[task]
    return torch.tensor([getattr(example, attribute) for example in examples], dtype=torch.long)


def inputs_for(examples: Sequence[SortExample]) -> torch.Tensor:
    return torch.tensor([example.inputs for example in examples], dtype=torch.long)


def shuffled_rows(
    inputs: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    order = torch.rand(inputs.shape, device=inputs.device, generator=generator).argsort(dim=1)
    return inputs.gather(1, order)


def decoder_inputs(targets: torch.Tensor, bos_id: int) -> torch.Tensor:
    bos = torch.full(
        (targets.shape[0], 1), bos_id, dtype=torch.long, device=targets.device
    )
    return torch.cat((bos, targets[:, :-1]), dim=1)


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
    squares = sum(parameter.detach().float().square().sum() for parameter in model.parameters())
    return math.sqrt(squares.item())


def batches(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


@torch.inference_mode()
def evaluate(
    model: SortTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype_name: str,
    strata: torch.Tensor | None = None,
) -> Metrics:
    model.eval()
    loss_sum = token_correct = in_set_correct = set_correct = exact_correct = 0.0
    token_count = inputs.numel()
    strata_correct = {name: 0 for name in STRATA}
    strata_counts = {name: 0 for name in STRATA}
    for selection in batches(len(inputs), batch_size):
        x = inputs[selection].to(device, non_blocking=device.type == "cuda")
        y = targets[selection].to(device, non_blocking=device.type == "cuda")
        with amp_context(device, dtype_name):
            logits = model(x, decoder_inputs(y, model.bos_id))
            loss_sum += F.cross_entropy(
                logits.flatten(0, 1), y.flatten(), reduction="sum"
            ).item()
            teacher = logits.argmax(dim=-1)
        token_correct += teacher.eq(y).sum().item()
        del logits, teacher
        with amp_context(device, dtype_name):
            generated = model.generate(x)
        in_set_correct += generated.unsqueeze(-1).eq(x.unsqueeze(1)).any(dim=-1).sum().item()
        exact_matches = generated.eq(y).all(dim=1)
        exact_correct += exact_matches.sum().item()
        set_correct += generated.sort(dim=1).values.eq(x.sort(dim=1).values).all(dim=1).sum().item()
        if strata is not None:
            batch_strata = strata[selection].to(device)
            for index, name in enumerate(STRATA):
                selected = batch_strata.eq(index)
                strata_counts[name] += selected.sum().item()
                strata_correct[name] += (exact_matches & selected).sum().item()
    return Metrics(
        loss_sum / token_count,
        token_correct / token_count,
        in_set_correct / token_count,
        set_correct / len(inputs),
        exact_correct / len(inputs),
        {
            name: (
                strata_correct[name] / strata_counts[name]
                if strata_counts[name] else None
            )
            for name in STRATA
        },
        strata_counts,
    )


def load_examples(
    args: argparse.Namespace,
) -> tuple[list[SortExample], list[SortExample], DatasetConfig]:
    if args.data:
        loaded = load_dataset(args.data)
        config = DatasetConfig(**loaded.metadata["config"])
        return (
            list(loaded.train.examples), list(loaded.test.examples),
            config,
        )
    config = DatasetConfig(
        n=args.n, m=args.m, train_percent=args.train_percent,
        modulus=args.modulus, seed=args.data_seed, n_test=args.n_test,
        enumerate_limit=args.enumerate_limit, train_count=args.train_count,
        split_strategy=args.split_strategy,
    )
    train_ranks, test_ranks, _ = split_ranks(config)
    return list(iter_examples(train_ranks, config)), list(iter_examples(test_ranks, config)), config


def subset_for_eval(
    inputs: torch.Tensor, targets: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 0 or count >= len(inputs):
        return inputs, targets
    return inputs[:count], targets[:count]


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
    """Prevent accidentally combining different trajectories in one CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"log CSV schema does not match this version: {path}")
        rows = list(reader)
    try:
        steps = [int(row["step"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot append to malformed log CSV: {path}") from exc
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"log CSV has duplicate or non-increasing steps: {path}")
    if resume_step is None:
        raise ValueError(f"log CSV already contains a run; choose a new path: {path}")
    if steps and steps[-1] > resume_step:
        raise ValueError(
            f"log CSV continues past resume step {resume_step}; choose a new path: {path}"
        )
    if any(row.get("run_signature_sha256") != run_signature_sha256 for row in rows):
        raise ValueError(f"log CSV belongs to a different run: {path}")


def dataset_fingerprint(
    train_examples: Sequence[SortExample], test_examples: Sequence[SortExample]
) -> str:
    digest = hashlib.sha256()
    for split, examples in ((b"train", train_examples), (b"test", test_examples)):
        digest.update(split)
        for example in examples:
            digest.update(repr(example.inputs).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def signature_digest(signature: dict[str, str]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def lr_for_step(args: argparse.Namespace, step: int) -> float:
    warmup_scale = min(1.0, step / max(1, args.warmup))
    progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    if args.lr_schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif args.lr_schedule == "linear":
        decay = 1.0 - progress
    else:
        decay = 1.0
    return args.lr * warmup_scale * decay


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    train_examples, test_examples, data_config = load_examples(args)
    vocab_size, set_size = data_config.n, data_config.m
    if not train_examples or not test_examples:
        raise ValueError("both train and test splits must contain at least one example")
    train_x, test_x = inputs_for(train_examples), inputs_for(test_examples)
    train_y = targets_for(train_examples, args.task)
    test_y = targets_for(test_examples, args.task)
    train_ranks = [combination_rank(example.inputs, vocab_size) for example in train_examples]
    comparison_information = compute_train_comparison_information(
        train_ranks, data_config, args.task
    )
    stratum_ids = {name: index for index, name in enumerate(STRATA)}
    test_strata = torch.tensor(
        [
            stratum_ids[classify_test_example(example, comparison_information)]
            for example in test_examples
        ],
        dtype=torch.long,
    )
    if device.type == "cuda":
        # Training repeatedly samples a small split, so keep it resident on the GPU.
        train_x, train_y = train_x.to(device), train_y.to(device)
        test_x, test_y = test_x.pin_memory(), test_y.pin_memory()
    run_signature = {
        "task": args.task,
        "output_constraint": args.output_constraint,
        "model_seed": str(args.seed),
        "dataset_sha256": dataset_fingerprint(train_examples, test_examples),
    }
    run_signature_sha256 = signature_digest(run_signature)

    config = ModelConfig(
        vocab_size=vocab_size, set_size=set_size, n_embd=args.n_embd,
        n_head=args.n_head, n_enc_layer=args.n_enc_layer, n_layer=args.n_layer,
        dropout=args.dropout, output_constraint=args.output_constraint,
        tie_embeddings=args.tie, init_std=args.init_std, init_scale=args.init_scale,
    )
    model = SortTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != asdict(config):
            raise ValueError("checkpoint model configuration does not match this run")
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError("checkpoint task or dataset does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if args.steps <= start_step:
            raise ValueError(
                f"steps must be greater than checkpoint step {start_step} when resuming"
            )

    executable: nn.Module = model
    if args.compile and device.type == "cuda":
        executable = torch.compile(model, mode=args.compile_mode, dynamic=False)
    scaler = None
    if device.type == "cuda" and args.dtype == "float16":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()
    if checkpoint is not None:
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "-"
    print(
        f"device={device} gpu={gpu} dtype={args.dtype if device.type == 'cuda' else 'float32'} "
        f"compile={args.compile and device.type == 'cuda'}"
    )
    print(
        f"task={args.task} constraint={args.output_constraint} "
        f"train={len(train_x):,} test={len(test_x):,} "
        f"C({vocab_size},{set_size}) | params={sum(p.numel() for p in model.parameters()):,}"
    )
    effective_batch_size = len(train_x) if args.batch_size == -1 else args.batch_size
    sampling_mode = "full-batch" if args.batch_size == -1 else "with-replacement"
    print(
        f"train batch={effective_batch_size:,} sampling={sampling_mode} "
        f"data_device={train_x.device} | eval_limit={args.n_eval:,} "
        f"eval_batch={args.eval_batch:,}"
    )
    if args.log_csv:
        validate_log_target(
            Path(args.log_csv), start_step if args.resume else None,
            run_signature_sha256,
        )

    eval_train_x, eval_train_y = subset_for_eval(train_x, train_y, args.n_eval)
    eval_test_x, eval_test_y = subset_for_eval(test_x, test_y, args.n_eval)
    eval_test_strata = test_strata[: len(eval_test_x)]
    if device.type == "cuda":
        eval_test_strata = eval_test_strata.to(device)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps({"model": asdict(config), "args": vars(args)}, indent=2, default=str),
            encoding="utf-8",
        )
    started = time.perf_counter()
    batch_size = effective_batch_size
    step_generator = torch.Generator(device=train_x.device)
    full_batch_indices = (
        torch.arange(len(train_x), device=train_x.device)
        if args.batch_size == -1 else None
    )
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        learning_rate = lr_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        step_generator.manual_seed(args.seed + step)
        if full_batch_indices is not None:
            batch_indices = full_batch_indices
        else:
            batch_indices = torch.randint(
                len(train_x), (batch_size,), device=train_x.device,
                generator=step_generator,
            )
        x = shuffled_rows(train_x[batch_indices], step_generator)
        y = train_y[batch_indices]
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.dtype):
            logits = executable(x, decoder_inputs(y, model.bos_id))
            loss = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(),
                label_smoothing=args.label_smoothing,
            )
        if scaler:
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

        should_eval = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_eval:
            train_metrics = evaluate(model, eval_train_x, eval_train_y, args.eval_batch, device, args.dtype)
            test_metrics = evaluate(
                model, eval_test_x, eval_test_y, args.eval_batch, device,
                args.dtype, eval_test_strata,
            )
            norm = parameter_norm(model)
            elapsed = time.perf_counter() - started
            print(
                f"step {step:7,d}/{args.steps:,d} lr {learning_rate:.2e} |w| {norm:.2f} | "
                f"train loss {train_metrics.loss:.4f} tok {train_metrics.token_acc:.3f} "
                f"exact {train_metrics.exact_acc:.3f} | test loss {test_metrics.loss:.4f} "
                f"tok {test_metrics.token_acc:.3f} exact {test_metrics.exact_acc:.3f} "
                f"in-set {test_metrics.gen_in_set_token_acc:.3f} "
                f"set {test_metrics.set_acc:.3f} | {elapsed:.1f}s"
            )
            strata_text = " ".join(
                f"{name}="
                f"{test_metrics.strata_exact_acc[name]:.3f}"
                if test_metrics.strata_exact_acc[name] is not None
                else f"{name}=-"
                for name in STRATA
            )
            print(f"  test strata exact: {strata_text}")
            if args.log_csv:
                write_csv(Path(args.log_csv), {
                    "step": step, "lr": learning_rate, "weight_norm": norm,
                    "train_loss": train_metrics.loss,
                    "train_token_acc": train_metrics.token_acc,
                    "train_gen_in_set_token_acc": train_metrics.gen_in_set_token_acc,
                    "train_set_acc": train_metrics.set_acc,
                    "train_exact_acc": train_metrics.exact_acc,
                    "test_loss": test_metrics.loss,
                    "test_token_acc": test_metrics.token_acc,
                    "test_gen_in_set_token_acc": test_metrics.gen_in_set_token_acc,
                    "test_set_acc": test_metrics.set_acc,
                    "test_exact_acc": test_metrics.exact_acc,
                    "elapsed_seconds": elapsed, "task": args.task,
                    "output_constraint": args.output_constraint,
                    "run_signature_sha256": run_signature_sha256,
                    "train_eval_count": len(eval_train_x),
                    "test_eval_count": len(eval_test_x),
                    **{
                        f"test_{name}_exact_acc": test_metrics.strata_exact_acc[name]
                        for name in STRATA
                    },
                    **{
                        f"test_{name}_count": test_metrics.strata_counts[name]
                        for name in STRATA
                    },
                })
        if out_dir and args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(
                out_dir / f"ckpt_{step:08d}.pt", model, optimizer, config,
                run_signature, scaler, step,
            )

    if out_dir:
        save_checkpoint(
            out_dir / "ckpt_final.pt", model, optimizer, config,
            run_signature, scaler, args.steps,
        )
    sample_count = min(5, len(test_x))
    generated = model.generate(test_x[:sample_count].to(device)).cpu()
    print("--- unseen combinations ---")
    for x, expected, actual in zip(test_x[:sample_count], test_y[:sample_count], generated):
        print(f"in {x.tolist()} -> {actual.tolist()} expected {expected.tolist()} {'OK' if torch.equal(actual, expected) else 'MISS'}")
    return 0


def save_checkpoint(
    path: Path,
    model: SortTransformer,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    run_signature: dict[str, str],
    scaler: object | None,
    step: int,
) -> None:
    torch.save({
        "step": step, "model_config": asdict(config), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "run_signature": run_signature,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--task", choices=TASKS, default="ascending")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--m", type=int, default=4)
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument("--train-count", type=int)
    sizing.add_argument("--train-percent", type=float)
    parser.add_argument(
        "--split-strategy", choices=("random", "relation-complete"), default="random"
    )
    parser.add_argument("--modulus", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--enumerate-limit", type=int, default=5_000_000)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-enc-layer", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tie", action="store_true")
    parser.add_argument(
        "--output-constraint", choices=OUTPUT_CONSTRAINTS, default="permutation"
    )
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="positive values sample exactly that many rows with replacement; -1 uses each train row once",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
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
    parser.add_argument(
        "--eval-batch", type=int, default=1024,
        help="evaluation chunk size; evaluation never repeats rows",
    )
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_embd <= 0 or args.n_head <= 0 or args.n_embd % args.n_head:
        raise ValueError("n-embd must be positive and divisible by n-head")
    if args.n_layer < 1 or args.n_enc_layer < 0:
        raise ValueError("n-layer must be positive and n-enc-layer nonnegative")
    if args.steps < 1 or args.batch_size == 0 or args.batch_size < -1:
        raise ValueError("steps must be positive and batch-size must be -1 or positive")
    if args.eval_every < 1 or args.eval_batch < 1:
        raise ValueError("evaluation intervals and batches must be positive")
    if args.n_eval == 0 or args.n_eval < -1:
        raise ValueError("n-eval must be -1 or positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be in [0, 1)")
    if args.label_smoothing and args.output_constraint != "free":
        raise ValueError("label smoothing requires --output-constraint free")
    if not args.data and args.train_count is None and args.train_percent is None:
        args.train_count = 128


def apply_smoke_settings(args: argparse.Namespace) -> None:
    args.data = None
    args.n, args.m, args.train_count, args.train_percent, args.n_test = 7, 3, 18, None, 17
    args.split_strategy = "random"
    args.n_embd, args.n_head = 32, 4
    args.n_enc_layer, args.n_layer = 1, 1
    args.steps, args.batch_size = 20, -1
    args.eval_every, args.n_eval, args.eval_batch = 10, -1, 128
    args.warmup, args.device, args.compile = 2, "cpu", False
    if args.log_csv is None:
        args.log_csv = Path(tempfile.gettempdir()) / "sortformer_smoke.csv"
        args.log_csv.unlink(missing_ok=True)


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
