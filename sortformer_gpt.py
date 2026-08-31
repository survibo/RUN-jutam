"""Train a causal decoder-only GPT on random-order sorting sequences.

The model consumes ``[random input][BOS][previous target tokens]`` with one
causal self-attention stack and predicts only the fixed-length target suffix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

import gpt_sortdata
import sortdata
import sortformer as base


@dataclass(frozen=True)
class GPTModelConfig(base.ModelConfig):
    architecture: str = "causal-gpt-v1"


class GPTBlock(nn.Module):
    def __init__(self, config: GPTModelConfig) -> None:
        super().__init__()
        self.attn_norm = base.RMSNorm(config.n_embd)
        self.attn = base.Attention(config.n_embd, config.n_head, config.dropout)
        self.mlp_norm = base.RMSNorm(config.n_embd)
        self.mlp = base.MLP(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(x)
        x = x + self.attn(normalized, normalized, causal_mask)
        return x + self.mlp(self.mlp_norm(x))


class GPTSortTransformer(base.SortTransformer):
    """Single-stack GPT with a fixed input/BOS/output serialization."""

    def __init__(self, config: GPTModelConfig) -> None:
        nn.Module.__init__(self)
        if config.n_enc_layer != 0:
            raise ValueError("causal GPT requires n_enc_layer=0")
        self.config = config
        self.bos_id = config.vocab_size
        self.value_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.bos_embedding = nn.Parameter(torch.empty(config.n_embd))
        # The longest context is m input tokens + BOS + (m - 1) targets.
        self.position_embedding = nn.Embedding(2 * config.set_size, config.n_embd)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.n_layer))
        self.final_norm = base.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        nn.init.normal_(self.bos_embedding, mean=0.0, std=config.init_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.value_embedding.weight
        if config.init_scale != 1.0:
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.mul_(config.init_scale)

    def _embed_decoder_inputs(self, decoder_inputs: torch.Tensor) -> torch.Tensor:
        values = self.value_embedding(decoder_inputs.clamp_max(self.bos_id - 1))
        return torch.where(
            decoder_inputs.eq(self.bos_id).unsqueeze(-1),
            self.bos_embedding.view(1, 1, -1),
            values,
        )

    def forward(
        self, inputs: torch.Tensor, decoder_inputs: torch.Tensor
    ) -> torch.Tensor:
        if inputs.ndim != 2 or decoder_inputs.ndim != 2:
            raise ValueError("inputs and decoder_inputs must be rank-2 tensors")
        if inputs.shape[0] != decoder_inputs.shape[0]:
            raise ValueError("inputs and decoder_inputs must have the same batch size")
        if inputs.shape[1] != self.config.set_size:
            raise ValueError(
                f"expected input length {self.config.set_size}, got {inputs.shape[1]}"
            )
        if (
            decoder_inputs.shape[1] < 1
            or decoder_inputs.shape[1] > self.config.set_size
        ):
            raise ValueError("decoder context length must be in [1, set_size]")
        input_length = inputs.shape[1]
        x = torch.cat(
            (self.value_embedding(inputs), self._embed_decoder_inputs(decoder_inputs)),
            dim=1,
        )
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        # Positions occupied by BOS/previous targets predict the target suffix.
        logits = self.lm_head(self.final_norm(x[:, input_length:]))
        if self.config.output_constraint != "free":
            logits = logits.masked_fill(
                ~self._valid_token_mask(inputs, decoder_inputs), float("-inf")
            )
        return logits


def load_examples(
    args: argparse.Namespace,
) -> tuple[
    list[gpt_sortdata.GPTSortExample],
    list[gpt_sortdata.GPTSortExample],
    sortdata.DatasetConfig,
]:
    if args.data:
        loaded = gpt_sortdata.load_dataset(args.data)
        config = sortdata.DatasetConfig(**loaded.metadata["config"])
        return list(loaded.train.examples), list(loaded.test.examples), config
    config = sortdata.DatasetConfig(
        n=args.n,
        m=args.m,
        train_percent=args.train_percent,
        modulus=args.modulus,
        seed=args.data_seed,
        n_test=args.n_test,
        enumerate_limit=args.enumerate_limit,
        train_count=args.train_count,
        split_strategy=args.split_strategy,
    )
    train_ranks, test_ranks, _ = sortdata.split_ranks(config)
    return (
        list(gpt_sortdata.iter_examples(train_ranks, config, "train")),
        list(gpt_sortdata.iter_examples(test_ranks, config, "test")),
        config,
    )


def _combination_rank(values: Sequence[int], n: int) -> int:
    return sortdata.combination_rank(tuple(sorted(values)), n)


def _classify_test_example(
    example: gpt_sortdata.GPTSortExample | Sequence[int],
    information: sortdata.TrainComparisonInformation,
) -> str:
    values = (
        example.inputs
        if isinstance(example, gpt_sortdata.GPTSortExample)
        else example
    )
    return sortdata.classify_test_example(tuple(sorted(values)), information)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    action = parser._option_string_actions["--n-enc-layer"]
    parser._remove_action(action)
    action.container._group_actions.remove(action)
    for option in action.option_strings:
        parser._option_string_actions.pop(option, None)
    parser.set_defaults(n_enc_layer=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        base.apply_smoke_settings(args)
        args.n_enc_layer = 0
    try:
        base.validate_args(args)
        # Reuse the mature optimizer/evaluation/checkpoint pipeline while swapping
        # only the architecture and random-order dataset adapter.
        base.ModelConfig = GPTModelConfig
        base.SortTransformer = GPTSortTransformer
        base.load_examples = load_examples
        base.combination_rank = _combination_rank
        base.classify_test_example = _classify_test_example
        return base.train(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
