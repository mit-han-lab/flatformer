import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from mmcv.runner import auto_fp16
from torch.nn import functional as F

from ..builder import MIDDLE_ENCODERS

__all__ = ["FlatFormer"]


def _create_cu_seqlens(batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        num_tokens * (batch_size + 1),
        step=num_tokens,
        dtype=torch.int32,
        device=device,
    )


class FlashAttention(nn.MultiheadAttention):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self._qkv_same_embed_dim

        batch_size, num_tokens, embed_dim = q.shape
        head_dim = embed_dim // self.num_heads

        x = torch.stack([q, k, v])
        x = x.view(3, -1, x.shape[-1])
        x = torch.baddbmm(self.ib(), x, self.iw())
        qkv = x.view(3, -1, self.num_heads, head_dim).transpose(0, 1)

        cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, qkv.device)
        x = flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, num_tokens, 0)
        x = x.view(batch_size, num_tokens, -1)

        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        return x, None

    def iw(self) -> torch.Tensor:
        tensor = self.in_proj_weight
        tensor = tensor.view(3, -1, tensor.shape[-1])
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor

    def ib(self) -> torch.Tensor:
        tensor = self.in_proj_bias
        tensor = tensor.view(3, 1, -1)
        return tensor


class GroupAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.attn = FlashAttention(in_channels, num_heads)

    def forward(self, x, pe):
        size = x.shape[0]
        num_groups = int(math.ceil(size / self.group_size))

        x = x.view(num_groups, self.group_size, -1)
        pe = pe.view(num_groups, self.group_size, -1)

        q = k = x + pe
        v = x
        x, _ = self.attn(q, k, v)

        x = x.view(num_groups * self.group_size, -1)

        return x


class BasicLayer(nn.Module):
    def __init__(self, in_channels, num_heads, activation, group_size) -> None:
        super().__init__()
        self.attn = GroupAttention(in_channels, num_heads, group_size)

        self.fc1 = nn.Linear(in_channels, 2 * in_channels)
        self.fc2 = nn.Linear(2 * in_channels, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.act = _get_activation_fn(activation)

        self.fp16_enabled = False

    @auto_fp16(apply_to=("src", "pe"))
    def forward(self, src, pe):
        src = self.norm1(src + self.attn(src, pe))
        src = self.norm2(src + self.fc2(self.act(self.fc1(src))))

        return src


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads,
        activation,
        group_size,
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList()
        for _ in range(4):
            layer = BasicLayer(
                in_channels,
                num_heads,
                activation,
                group_size=group_size,
            )
            self.block.append(layer)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, mappings: Dict[str, Any]) -> torch.Tensor:
        for k, name in enumerate(["x", "x_shift", "y", "y_shift"]):
            indices = mappings[name]
            x[indices] = self.block[k](x[indices][mappings["flat2win"]], pe[indices][mappings["flat2win"]])[
                mappings["win2flat"]
            ]

        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu


@torch.inference_mode()
def get_window_coors_shift(coords, sparse_shape, window_shape, shifted):
    n, m, _ = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    if shifted:
        shift_x, shift_y = (n2 // 2, m2 // 2)
        x = coords[:, 3] + shift_x
        y = coords[:, 2] + shift_y
    else:
        x = coords[:, 3]
        y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
        self,
        window_shape,
        sparse_shape,
        group_size,
    ) -> None:
        super().__init__()
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.group_size = group_size

    def forward(self, coords: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        coords = coords.long()

        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
            torch.div(
                batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                self.group_size,
                rounding_mode="trunc",
            )
            * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1]).to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1]).to(coords.device)
        for i in range(batch_size):
            win2flat[batch_start_indices[i] : batch_start_indices[i + 1]] += (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
            if num_per_batch[i] != num_per_batch_p[i]:
                flat2win[
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                    - self.group_size
                ]
            flat2win[batch_start_indices_p[i] : batch_start_indices_p[i + 1]] -= (
                batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}
        for shifted in [False, True]:
            (
                n2,
                m2,
                n1,
                m1,
                x1,
                y1,
                x2,
                y2,
            ) = get_window_coors_shift(coords, self.sparse_shape, self.window_shape, shifted=shifted)
            vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
            vx += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
            vy += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
            _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        return mappings


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        feat_dim,
        sparse_shape,
        normalize_pos,
        pos_temperature,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.sparse_shape = sparse_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature

    def forward(self, coors, dtype):
        size_x, size_y, size_z = self.sparse_shape

        x, y = coors[:, 3], coors[:, 2]

        if self.normalize_pos:
            x = x / size_x * 2 * 3.1415  # [-pi, pi]
            y = y / size_y * 2 * 3.1415  # [-pi, pi]

        inv_freq = self.inv_freq

        # [num_tokens, pos_length]
        pex = x[:, None] / inv_freq()[None, :]
        pey = y[:, None] / inv_freq()[None, :]

        # [num_tokens, pos_length]
        pex = torch.stack([pex[:, ::2].sin(), pex[:, 1::2].cos()], dim=-1).flatten(1)
        pey = torch.stack([pey[:, ::2].sin(), pey[:, 1::2].cos()], dim=-1).flatten(1)
        pe = torch.cat([pex, pey], dim=-1).to(dtype)

        gap = self.feat_dim - pe.size(1)
        if gap > 0:
            pe_p = torch.zeros((pe.size(0), gap), dtype=dtype, device=coors.device)
            pe = torch.cat([pe, pe_p], dim=1)

        return pe

    def inv_freq(self):
        ndim = 2
        pos_length = (self.feat_dim // (ndim * 2)) * 2

        # [pos_length]
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device="cuda")
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        return inv_freq


@MIDDLE_ENCODERS.register_module()
class FlatFormer(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_heads=8,
        num_blocks=2,
        activation="gelu",
        window_shape=(9, 9, 1),
        sparse_shape=(468, 468, 1),
        output_shape=(468, 468),
        pos_temperature=10000,
        normalize_pos=False,
        group_size=69,
    ) -> None:
        super().__init__()
        self.group_size = group_size

        self.embedding = PositionalEmbedding(in_channels, sparse_shape, normalize_pos, pos_temperature)
        self.mapping = FlattenedWindowMapping(
            window_shape=window_shape,
            sparse_shape=sparse_shape,
            group_size=group_size,
        )

        self.block_list = nn.ModuleList()
        for _ in range(num_blocks):
            self.block_list.append(BasicBlock(in_channels, num_heads, activation, group_size))

        self._reset_parameters()

        self.output_shape = output_shape

    @auto_fp16(apply_to=('x',))
    def forward(self, x, coords, batch_size):
        pe = self.embedding(coords, x.dtype)
        mappings = self.mapping(coords, batch_size)

        for _, block in enumerate(self.block_list):
            x = block(x, pe, mappings)

        x = self.recover_bev(x, coords, batch_size)

        return x

    def _reset_parameters(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(feat_dim, nx * ny, dtype=voxel_feat.dtype, device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :]  # [n, c]
            voxels = voxels.t()  # [c, n]
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas
