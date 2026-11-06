import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


def closest_square_factors(N):
    s = int(math.isqrt(N)) 
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i


class ExpLinearAfterThreshold(nn.Module):
    def __init__(self, max_val=0.0):
        super(ExpLinearAfterThreshold, self).__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.exp(x.clamp(max=self.max_val)) + torch.relu(x - self.max_val)


class SepConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        act_layer=nn.ReLU,
        bias=True,
    ):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


class ConvProjection(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        kernel_size=3,
        q_stride=1,
        k_stride=1,
        v_stride=1,
        dropout=0.0,
        last_stage=False,
        bias=True,
    ):

        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(in_channels=dim, out_channels=inner_dim, kernel_size=kernel_size, stride=q_stride, padding=pad,bias=bias)
        self.to_k = SepConv2d(in_channels=dim, out_channels=inner_dim, kernel_size=kernel_size, stride=k_stride, padding=pad,bias=bias)
        self.to_v = SepConv2d(in_channels=dim, out_channels=inner_dim, kernel_size=kernel_size, stride=v_stride, padding=pad,bias=bias)

    def forward(self, x, attn_kv=None):
        h = self.heads
        attn_kv = x if attn_kv is None else attn_kv

        q = self.to_q(x)
        q = rearrange(q, "b (h d) l w -> b h (l w) d", h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, "b (h d) l w -> b h (l w) d", h=h)
        v = rearrange(v, "b (h d) l w -> b h (l w) d", h=h)
        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is None:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(
            B_, N, 1, self.heads, C // self.heads
        ).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(
            B_, N_kv, 2, self.heads, C // self.heads
        ).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    # print('rope')
    return (x * cos) + (rotate_every_two(x) * sin)


class PnPNystraAttention(nn.Module):
    def __init__(
        self,
        num_landmarks,
        iters,
        dim,
        num_heads,
        win_size=7,
        token_projection="conv",
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.token_projection = token_projection
        if token_projection == "conv":
            self.qkv = ConvProjection(
                dim, num_heads, dim // num_heads, bias=qkv_bias
            )
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.activ = ExpLinearAfterThreshold()

        self.num_landmarks = num_landmarks
        self.iters = iters

        self.RoPE = RoPE(dim, num_heads)

    def moore_penrose_iter_pinv(self, x, iters=6):
        device = x.device

        abs_x = torch.abs(x)
        col = abs_x.sum(dim=-1)
        row = abs_x.sum(dim=-2)
        z = rearrange(x, "... i j -> ... j i") / (
            torch.max(col) * torch.max(row)
        )

        I = torch.eye(x.shape[-1], device=device)
        I = rearrange(I, "i j -> () i j")

        for _ in range(iters):
            xz = x @ z
            z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

        return z
    
    def forward(self, x, attn_kv=None):
        B_, C ,H, W = x.shape
        sin, cos = self.RoPE((H, W))
        N = H*W
        if attn_kv is not None:
            _,_,H1,W1 = attn_kv.shape
            if H1*W1!=N:
                attn_kv = F.avg_pool2d(attn_kv, kernel_size=2, stride=2)
            if self.token_projection == "linear":
                attn_kv = rearrange(attn_kv, "b c h w -> b (h w) c")

        if self.token_projection == "linear":
            x = rearrange(x, "b c h w -> b (h w) c")

        num_landmarks = self.num_landmarks
        iters = self.iters

        q, k, v = self.qkv(x,attn_kv)
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)
        q = q * self.scale

        h, w = closest_square_factors(num_landmarks)

        q_m = F.adaptive_avg_pool2d(
            q.reshape(B_ * self.num_heads, H, W, self.dim // self.num_heads)
            .permute(0, 3, 1, 2),
            output_size=(h, w),
        ).permute(0, 2, 3, 1).reshape(
            B_, self.num_heads, num_landmarks, self.dim // self.num_heads
        )
        k_m = F.adaptive_avg_pool2d(
            k.reshape(B_ * self.num_heads, H, W, self.dim // self.num_heads)
            .permute(0, 3, 1, 2),
            output_size=(h, w),
        ).permute(0, 2, 3, 1).reshape(
            B_, self.num_heads, num_landmarks, self.dim // self.num_heads
        )

        temp = self.activ(q_m @ k_m.transpose(-2, -1))

        pseudo_inv = self.moore_penrose_iter_pinv(temp, iters)

        prod = (
            self.activ(q @ k_m.transpose(-2, -1))
            @ pseudo_inv
            @ (
                self.activ(q_m @ k.transpose(-2, -1))
                @ torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
            )
        )

        x = (
            prod[..., :-1] / (prod[..., -1].unsqueeze(-1) + 1e-12)
        ).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.reshape(B_, C, H, W)
