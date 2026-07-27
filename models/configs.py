"""PLACEHOLDER -- model configurations for gates and benchmarks.

``gate_tiny`` (design doc step 0) -- small enough that brute-force per-objective
autograd is cheap, large enough to exercise every layer type::

    n_layer=2, n_head=2, n_embd=64, block_size=16, vocab_size=65,
    dropout=0.0, m=4 per-sequence mean-CE objectives, fp64

Two non-negotiables:
  * ``dropout = 0.0`` -- otherwise the brute-force pass and the hooked pass see
    different networks and the gate compares nothing;
  * gates run with ``bias=True`` **and** ``bias=False``, since the bias terms
    are separate summands in every identity.

fp64 for gates, dtype configurable for runs (A5000 fp64 throughput is
rate-limited, so real-shape runs are fp32).

``nanogpt_small`` and the nanochat/Qwen target shapes belong here too, for
``bench/`` to import.
"""

from __future__ import annotations
import torch 
import torch.nn as nn
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
    tie_weights: bool = False   # False for 5a-5d
    m: int = 4                  # objectives = batch size

    def cfg2gpt(self) -> GPTConfig:
        assert self.dropout == .0
        return GPTConfig(
            n_layer=self.n_layer,
            n_head = self.n_head,
            n_embd= self.n_embd,
            block_size=self.block_size,
            vocab_size= self.vocab_size,
            dropout= self.dropout,
            bias= self.bias
        )

class gateGPT(GPT):
   def _init__(self, config: GPTConfig, tie_weights: bool = False):
      super().__init__(config)
      if not tie_weights:
         self.transformer.wte.weight = nn.Parameter(
            self.transformer.wte.weight.detach().clone()
         )


def build_gate_model(bias: bool, tie_weights: bool = False, device="cuda", dtype=torch.float64) -> GPT:
  cfg = GateTinyConfig(bias = bias, tie_weights=tie_weights)
  model = gateGPT(cfg.cfg2gpt() , tie_weights)
  model.to(device=device, dtype=dtype)
  model.eval()
  return model

    
def gate_batch(config: GateTinyConfig, seed : int, device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device = device).manual_seed(seed)
    idx = torch.randint(0, config.vocab_size, (config.m, config.block_size), generator=g, device=device)
    targets = torch.randint(0, config.vocab_size, (config.m, config.block_size), generator=g, device=device)
    return idx, targets
def forward_logits(model, idx) -> torch.Tensor:               # [B,T,V], no scalar loss shortcut
  logits, _ = model(idx, targets=None)
  return logits
  
def per_sequence_losses(logits, targets) -> torch.Tensor:     # [m]
  B, T, V = logits.shape

  per_token_loss=  torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")

  per_token_loss = per_token_loss.reshape(B,T)
  per_sequence_loss = per_token_loss.mean(dim=1)
  return per_sequence_loss