"""
algorithm3, corrected. Four issues found vs. the original, verified by crash
or by direct comparison against torchjd.autogram.Engine.compute_gramian:

1. Loss-seed mismatch. `A = (out - y) * 2` is the MSE-loss seed, assuming y is
   the same shape as out. The benchmark uses CrossEntropyLoss with integer
   class labels -- shapes don't even match ([m,10] vs [m]), so this crashes
   immediately (confirmed). Fixed: for CE loss, d(loss_i)/d(logit_i) has the
   closed form softmax(logit_i) - one_hot(y_i). This is THE seed for whatever
   loss you're using -- if you switch losses later, this line is what you
   change, nothing else.

2. Missing nn.MaxPool2d branch. The if/elif chain has no case for it, so a
   MaxPool2d layer is silently skipped -- A passes through unmodified instead
   of being routed back through the max positions. Silent, not a crash: it
   quietly produces a wrong-but-plausible-looking Gramian. Fixed via
   F.max_pool2d(..., return_indices=True) recomputed from the cached input,
   then F.max_unpool2d to scatter A back through the winning positions.

3. Conv2d never propagates A backward through itself. The Linear branch ends
   with `A = A @ layer.weight` so earlier layers see the right gradient; the
   Conv2d branch computes this layer's own weight-gradient contribution
   correctly, then just... stops. Every layer processed after (earlier in the
   network, e.g. an upstream Conv2d/MaxPool/ELU) silently receives a stale A
   that never passed through this conv's weights. On the paper's 3-conv
   architecture this corrupts every Gramian contribution before the last
   conv layer. Fixed with torch.nn.grad.conv2d_input -- a PyTorch-native,
   groups-aware primitive that computes exactly d(loss)/d(input) given a
   batched d(loss)/d(output); used here per-example (each row of A is
   independent, so a batched call is just a correct per-example vmap already,
   no cross-example mixing).

4. The weight-gradient Gramian term itself (the unfold+matmul Hadamard trick)
   assumes groups=1. Two of the three conv layers in the paper's CNN are
   grouped/depthwise (groups=32*w and groups=64*w). A plain unfold ignores
   groups and mixes input/output channels that aren't actually connected by
   any real weight, inflating the Gramian with terms that don't exist. Fixed
   with an explicit per-group loop below -- correct, not yet the fully
   vectorized version; see the note at the bottom for how to vectorize once
   this passes the correctness gate.
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
        cache[l] = out.clone()
        out = l(out)

    # Fix 1: CE-loss-correct seed. d(CE_i)/d(logit_i) = softmax(logit_i) - onehot(y_i)
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
            elu_derivative = torch.where(z > 0, torch.ones_like(z), torch.exp(z))
            A = A * elu_derivative

        elif isinstance(layer, nn.MaxPool2d):  # Fix 2
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
            Cout, Cin = layer.out_channels, layer.in_channels
            Cout_g, Cin_g = Cout // G, Cin // G

            # Fix 4: per-group weight-gradient Gramian contribution (correctness
            # baseline -- see file docstring for the vectorized follow-up).
            for g in range(G):
                x_in_g = x_in[:, g * Cin_g:(g + 1) * Cin_g]
                A_g = A[:, g * Cout_g:(g + 1) * Cout_g]
                unfolded_g = F.unfold(
                    x_in_g, kernel_size=layer.kernel_size,
                    padding=layer.padding, stride=layer.stride,
                )
                unfolded_A_g = A_g.reshape(m, Cout_g, -1)
                J_W_g = (unfolded_A_g @ unfolded_g.transpose(1, 2)).reshape(m, -1)
                gramian += (J_W_g @ J_W_g.T).double()

            if layer.bias is not None:
                temp = A.sum(dim=(2, 3))
                gramian += (temp @ temp.T).double()

            # Fix 3: propagate A backward through this conv's weights.
            A = torch.nn.grad.conv2d_input(
                x_in.shape, layer.weight, A,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups,
            )

        elif isinstance(layer, nn.Flatten):
            x_in = cache[layer]
            A = A.reshape(x_in.shape)

    return gramian


# Vectorization note for Fix 4, once this passes the correctness gate:
# reshape unfolded to [m, G, Cin_g*kH*kW, L] and unfolded_A to [m, G, Cout_g, L],
# then a single batched matmul over the folded (m*G) leading dims replaces the
# Python loop over G -- turns G separate small matmuls (and G separate kernel
# launches) into one batched call. Re-verify against this file after that change;
# don't trust the vectorized version until it matches this one to the same tolerance.