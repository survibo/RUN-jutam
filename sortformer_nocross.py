"""Train a decoder-only prefix Transformer on set-to-permutation sorting tasks."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import torch.nn as nn

import sortformer as base


class PrefixDecoderBlock(nn.Module):
    def __init__(self, config: base.ModelConfig) -> None:
        super().__init__()
        self.attn_norm = base.RMSNorm(config.n_embd)
        self.attn = base.Attention(config.n_embd, config.n_head, config.dropout)
        self.mlp_norm = base.RMSNorm(config.n_embd)
        self.mlp = base.MLP(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(x)
        x = x + self.attn(normalized, normalized, mask)
        return x + self.mlp(self.mlp_norm(x))


class PrefixSortTransformer(base.SortTransformer):
    """Decoder-only model that treats the unordered input set as a prefix."""

    def __init__(self, config: base.ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.bos_id = config.vocab_size
        self.input_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.decoder_embedding = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.decoder_position = nn.Embedding(config.set_size, config.n_embd)
        self.decoder = nn.ModuleList(
            PrefixDecoderBlock(config) for _ in range(config.n_layer)
        )
        self.decoder_norm = base.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.input_embedding.weight
        if config.init_scale != 1.0:
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.mul_(config.init_scale)

    @staticmethod
    def _prefix_mask(
        input_length: int, output_length: int, device: torch.device
    ) -> torch.Tensor:
        total_length = input_length + output_length
        mask = torch.ones(
            total_length, total_length, dtype=torch.bool, device=device
        )
        mask[:input_length, :input_length] = False
        mask[input_length:, :input_length] = False
        mask[input_length:, input_length:] = torch.triu(
            torch.ones(
                output_length, output_length, dtype=torch.bool, device=device
            ),
            diagonal=1,
        )
        return mask

    def forward(self, inputs: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        output_length = decoder_inputs.shape[1]
        positions = torch.arange(output_length, device=inputs.device)
        input_prefix = self.input_embedding(inputs)
        output_prefix = (
            self.decoder_embedding(decoder_inputs) + self.decoder_position(positions)
        )
        x = torch.cat((input_prefix, output_prefix), dim=1)
        mask = self._prefix_mask(inputs.shape[1], output_length, inputs.device)
        for block in self.decoder:
            x = block(x, mask)
        logits = self.lm_head(self.decoder_norm(x[:, inputs.shape[1] :]))
        if self.config.output_constraint != "free":
            logits = logits.masked_fill(
                ~self._valid_token_mask(inputs, decoder_inputs), float("-inf")
            )
        return logits


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
        base.SortTransformer = PrefixSortTransformer
        return base.train(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
