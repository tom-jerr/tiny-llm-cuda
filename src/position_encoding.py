import torch
import torch.nn as nn
from functools import lru_cache


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        max_seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        super().__init__()
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.half_dims = dims // 2
        self.max_seq_len = max_seq_len
        self.traditional = traditional
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs: torch.Tensor = torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)  # [seq_len, head_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        B, S, H, D = x.shape
        if offset is None:
            # from 0 to s-1th: [S, D] -> [1, S, D]
            cos_sin = self.cos_sin_cache[:S].unsqueeze(0).to(x.device)
        elif isinstance(offset, slice):
            # just one slice: [S, D] -> [1, S, D]
            assert (
                offset.stop - offset.start == S
            ), "Offset slice length must match sequence length"
            cos_sin = self.cos_sin_cache[offset].unsqueeze(0).to(x.device)

        elif isinstance(offset, list):
            # slice list, collase all cos and sin to one tensor: [S,D]->[B,S,D]
            assert (
                len(offset) == B
            ), "Number of slices in offset list must match batch size"
            cos_sin = torch.stack([self.cos_sin_cache[s] for s in offset], dim=0).to(
                x.device
            )

        else:
            raise TypeError(f"Unsupported type for offset: {type(offset)}")

        # [B, S, D/2] or [1, S, D/2]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.traditional:
            x = x.reshape(B, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            # Qwen2 style
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
        # [B, S, D/2] -> [B, S, 1, D/2]
        cos = cos.reshape(-1, S, 1, self.half_dims)
        sin = sin.reshape(-1, S, 1, self.half_dims)
        # [B, S, H, D/2]
        real = x1 * cos - x2 * sin
        imag = x1 * sin + x2 * cos
        if self.traditional:
            y = torch.stack([real, imag], dim=-1)
            y = y.reshape(B, S, H, D)
        else:
            y = torch.cat((real, imag), dim=-1)
            y = y.reshape(B, S, H, D)
        return y.type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.ndim == 2:
        # [seq_len, half_dim] case
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        # [B, seq_len, half_dim] case
        assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1])
        shape = [
            d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
        ]

    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: list[slice] | slice | None = None,
) -> torch.Tensor:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    B, seq_len = xq.shape[0], xq.shape[1]

    if offset is None:
        # from 0 to seq_len-1th
        freqs_cis = freqs_cis[:seq_len]
    elif isinstance(offset, slice):
        # just one slice
        assert (
            offset.stop - offset.start == seq_len
        ), "Offset slice length must match sequence length"
        freqs_cis = freqs_cis[offset]
    elif isinstance(offset, list):
        # slice list for each batch
        assert len(offset) == B, "Number of slices in offset list must match batch size"
        # 为每个batch应用对应的slice
        freqs_cis_list = [freqs_cis[s] for s in offset]
        freqs_cis = torch.stack(freqs_cis_list, dim=0)  # [B, seq_len, half_dim]
    else:
        raise TypeError(f"Unsupported type for offset: {type(offset)}")

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def apply_rotary_emb_qwen2(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: list[slice] | slice | None = None,
) -> torch.Tensor:
    B, seq_len = xq.shape[0], xq.shape[1]
    half_dim = xq.shape[-1] // 2

    # 分为前半部分和后半部分
    x1 = xq[..., :half_dim]  # 前半部分
    x2 = xq[..., half_dim:]  # 后半部分

    # 根据offset参数截取频率张量
    if offset is None:
        # from 0 to seq_len-1th
        freqs_cis = freqs_cis[:seq_len]
    elif isinstance(offset, slice):
        # just one slice
        assert (
            offset.stop - offset.start == seq_len
        ), "Offset slice length must match sequence length"
        freqs_cis = freqs_cis[offset]
    elif isinstance(offset, list):
        # slice list for each batch
        assert len(offset) == B, "Number of slices in offset list must match batch size"
        # 为每个batch应用对应的slice
        freqs_cis_list = [freqs_cis[s] for s in offset]
        freqs_cis = torch.stack(freqs_cis_list, dim=0)  # [B, seq_len, half_dim]
    else:
        raise TypeError(f"Unsupported type for offset: {type(offset)}")

    # 获取 cos 和 sin 分量
    cos_freqs = freqs_cis.real  # [seq_len, half_dim] or [B, seq_len, half_dim]
    sin_freqs = freqs_cis.imag  # [seq_len, half_dim] or [B, seq_len, half_dim]

    # reshape for broadcast
    if cos_freqs.ndim == 2:
        # [seq_len, half_dim] -> [1, seq_len, 1, half_dim]
        cos_freqs = cos_freqs.view(1, seq_len, 1, half_dim)
        sin_freqs = sin_freqs.view(1, seq_len, 1, half_dim)
    else:
        # [B, seq_len, half_dim] -> [B, seq_len, 1, half_dim]
        cos_freqs = cos_freqs.view(B, seq_len, 1, half_dim)
        sin_freqs = sin_freqs.view(B, seq_len, 1, half_dim)

    # 应用旋转
    # output[0:half_dim] = x1 * cos - x2 * sin
    # output[half_dim:dim] = x1 * sin + x2 * cos
    real = x1 * cos_freqs - x2 * sin_freqs
    imag = x1 * sin_freqs + x2 * cos_freqs

    # 重新组合
    xq_out = torch.cat([real, imag], dim=-1)
    return xq_out.type_as(xq)
