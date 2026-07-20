"""
algorithm3: identical to algo3_fixed.algorithm3 except the Conv2d
weight-gradient Gramian term is computed with ONE batched einsum over all
groups instead of a Python loop over G.

Derivation of the vectorized form (why this is the same math):
  per-group loop computed, for each g:
      J_g = A_g_reshaped @ unfolded_g^T          [m, Cout_g, Cin_g*kH*kW]
      gramian += flatten(J_g) @ flatten(J_g)^T
  Summing (J_g @ J_g^T) over g is exactly the Frobenius inner product of the
  full per-example weight-Jacobian rows, which is what one gets by stacking
  the per-group J_g blocks into J = [m, G*Cout_g*Cin_g*kH*kW] and doing a
  single J @ J^T. The einsum 'mgol,mgil->mgoi' does all G small matmuls as
  one batched kernel launch instead of G launches from Python.

Memory: identical to the loop version -- the [m, P_layer] block J is
materialized either way (grouped-conv P_layer is small: Cout * Cin_g * kH*kW).
The win is kernel-launch count and Python overhead, i.e. exactly the
dispatch-bound regime the Perfetto traces show.
"""
import torch
import torch.nn.functional as F
from torch import nn


def algorithm3(model, x, y, num_classes: int | None = None) -> torch.Tensor:
    m = x.shape[0]
    device = x.device
    layers = list(model.children())

    cache = {}
    out = x
    for l in layers:
        cache[l] = out
        out = l(out)

    C = num_classes or out.shape[-1]
    A = torch.softmax(out, dim=-1) - F.one_hot(y, num_classes=C).to(out.dtype)

    gramian = torch.zeros(m, m, dtype=torch.float64, device=device)

    for layer in reversed(layers):
        if isinstance(layer, nn.Linear):
            x_in = cache[layer]
            gramian += (A @ A.T).double() * (x_in @ x_in.T).double()
            if layer.bias is not None:
                gramian += (A @ A.T).double()
            A = A @ layer.weight

        elif isinstance(layer, nn.ELU):
            z = cache[layer]
            A = A * torch.where(z > 0, torch.ones_like(z), torch.exp(z))

        elif isinstance(layer, nn.MaxPool2d):
            x_in = cache[layer]
            _, indices = F.max_pool2d(
                x_in, kernel_size=layer.kernel_size, stride=layer.stride,
                padding=layer.padding, return_indices=True,
            )
            A = F.max_unpool2d(
                A, indices, kernel_size=layer.kernel_size, stride=layer.stride,
                padding=layer.padding, output_size=x_in.shape[-2:],
            )

        elif isinstance(layer, nn.Conv2d):
            x_in = cache[layer]
            G = layer.groups
            Cout = layer.out_channels
            Cout_g = Cout // G

            # --- vectorized weight-gradient Gramian term (was: loop over G) ---
            unf = F.unfold(x_in, 
                           kernel_size=layer.kernel_size,
                           padding=layer.padding,
                             stride=layer.stride)   # [m, Cin*kk, L]
            L = unf.shape[-1]
            unf = unf.view(m, G, -1, L)                                  # [m, G, Cin_g*kk, L]
            A_r = A.reshape(m, G, Cout_g, L)                             # [m, G, Cout_g, L]
            J = (A_r @ unf.transpose(-1, -2)).reshape(m, -1)
            gramian += (J @ J.T).double()

            if layer.bias is not None:
                temp = A.sum(dim=(2, 3))
                gramian += (temp @ temp.T).double()

            A = torch.nn.grad.conv2d_input(
                x_in.shape, layer.weight, A,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups,
            )

        elif isinstance(layer, nn.Flatten):
            A = A.reshape(cache[layer].shape)

    return gramian, out