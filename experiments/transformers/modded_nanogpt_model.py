"""Portable Track 3 model and Muon, adapted from Keller Jordan (MIT).

Reference: modded_nanogpt_reference/train_gpt_simple.py and its LICENSE.
The default architecture, initialization, and Muon math match that snapshot.
"""

import torch
from torch import nn
import torch.nn.functional as F

UPSTREAM_REVISION = "4ea6b937337a4889b8cfe3f38a93d120048d8f71"


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([
            angular_freq, angular_freq.new_zeros(dim // 4),
        ]))

    def forward(self, x):
        pos = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.proj = Linear(dim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x):
        b, t = x.shape[:2]
        q = self.q(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k(x).view(b, t, self.num_heads, self.head_dim)
        v = self.v(x).view(b, t, self.num_heads, self.head_dim)
        q = self.rotary(F.rms_norm(q, (self.head_dim,)))
        k = self.rotary(F.rms_norm(k, (self.head_dim,)))
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=0.12, is_causal=True,
        ).transpose(1, 2).contiguous().view(b, t, -1)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x):
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size=50304, num_layers=12, model_dim=768,
                 head_dim=128, compute_dtype=torch.float32, *,
                 head_init_std=0.0, fp32_embedding=False):
        super().__init__()
        if model_dim % head_dim or head_dim % 4:
            raise ValueError("model_dim must divide into heads; head_dim must be a multiple of 4")
        self.compute_dtype = compute_dtype
        self.head_init_std = head_init_std
        self.embed = nn.Embedding(vocab_size, model_dim).to(dtype=compute_dtype)
        self.blocks = nn.ModuleList([Block(model_dim, head_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)  # deliberately untied
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.reset_parameters()
        # Retain the reference's initial BF16 values, but accumulate small
        # optimizer updates in FP32. Forward activations remain compute_dtype.
        if fp32_embedding:
            self.embed.float()

    @torch.no_grad()
    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name.endswith("weight"):
                if name == "proj.weight" and self.head_init_std > 0:
                    p.normal_(std=self.head_init_std)
                elif "proj" in name:
                    p.zero_()
                elif "embed" in name:
                    p.normal_()
                else:
                    p.normal_(std=0.33**0.5 / p.size(-1)**0.5)
            elif name.endswith("bias"):
                p.zero_()
            elif name.endswith("gains"):
                p.fill_(1)
            else:
                raise ValueError(f"Uninitialized parameter: {name}")

    def forward(self, inputs):
        x = self.norm1(self.embed(inputs).to(dtype=self.compute_dtype))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()


def zeropower(gradient):
    # BF16 matches upstream on CUDA; FP32 is portable on MPS/CPU.
    x = gradient.to(torch.bfloat16 if gradient.device.type == "cuda" else torch.float32)
    transpose = gradient.size(-2) > gradient.size(-1)
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        a = x @ x.mT
        b = -1.5 * a + 0.5 * a @ a
        x = 2 * x + b @ x
    return x.mT if transpose else x


class Muon(torch.optim.Optimizer):
    """Single-device version of the tuned reference's hidden-matrix optimizer."""
    def __init__(self, params, lr=0.025, weight_decay=0.05, mu=0.95):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, mu=mu))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                momentum = state["momentum"]
                momentum.lerp_(p.grad, 1 - group["mu"])
                update = zeropower(p.grad.lerp(momentum, group["mu"]))
                update *= max(1, p.size(-2) / p.size(-1))**0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
