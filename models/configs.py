from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nanogpt.model import GPT, GPTConfig


@dataclass
class GateTinyConfig:
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 16
    vocab_size: int = 65
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = False  # False for gates 5a-5d
    m: int = 4  # objectives = batch size

    def cfg2gpt(self) -> GPTConfig:
        assert self.dropout == 0.0
        return GPTConfig(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            bias=self.bias,
        )


class GateGPT(GPT):
    """nanoGPT with optional weight tying disabled for gates 5a-5d."""

    def __init__(self, config: GPTConfig, *, tie_weights: bool = True):
        super().__init__(config)
        if not tie_weights:
            # GPT ties wte.weight to lm_head.weight; clone wte to break the link.
            self.transformer.wte.weight = nn.Parameter(
                self.transformer.wte.weight.detach().clone()
            )


def build_gate_model(
    bias: bool,
    tie_weights: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> GateGPT:
    cfg = GateTinyConfig(bias=bias, tie_weights=tie_weights)
    model = GateGPT(cfg.cfg2gpt(), tie_weights=tie_weights)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def gate_batch(
    config: GateTinyConfig,
    seed: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randint(
        0,
        config.vocab_size,
        (config.m, config.block_size),
        generator=g,
        device=device,
    )
    targets = torch.randint(
        0,
        config.vocab_size,
        (config.m, config.block_size),
        generator=g,
        device=device,
    )
    return idx, targets


def forward_logits(
    model: GPT, idx: torch.Tensor, *, batched_positions: bool = True
) -> torch.Tensor:
    """Return [B, T, V] logits without reducing to a scalar loss.

    ``batched_positions`` expands the position vector from ``[T]`` to ``[B, T]``.
    The logits are identical either way -- ``wpe([T])`` broadcasts to the same
    values -- but the *shape of wpe's output* decides whether the module is
    batched on dim 0, and that decides which reverse drivers can be used:

    * ``[B, T, d]`` (batched): every hooked module leads with the objective
      dimension, so one ones-seeded backward recovers every objective's upstream
      gradient (``driver="squashed"``). TorchJD's autogram engine requires the
      same thing for ``batch_dim=0``.
    * ``[T, d]`` (broadcast): autograd's broadcast-backward sums over the batch
      before the hook sees the gradient, so per-objective gradients survive only
      if one objective was seeded per pass. That restricts you to ``"loop"`` or
      ``"batched"``.

    Set it False only to exercise the unbatched path.
    """
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    if batched_positions:
        pos = pos.unsqueeze(0).expand(b, t)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def per_sequence_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy per sequence; one scalar objective per batch row."""
    b, t, v = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(-1, v),
        targets.reshape(-1),
        reduction="none",
    ).reshape(b, t)
    return per_token.mean(dim=1)
