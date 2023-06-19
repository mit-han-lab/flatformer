# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from torch import nn

__all__ = ["SECOND"]


@BACKBONES.register_module()
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    DEFAULT_NORM_CFG = {"type": "naiveSyncBN2d", "eps": 1e-3, "momentum": 0.01}
    DEFAULT_CONV_CFG = {"type": "Conv2d", "bias": False}

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: Sequence[int] = (128, 128, 256),
        layer_nums: Sequence[int] = (3, 5, 5),
        layer_strides: Sequence[int] = (2, 2, 2),
        norm_cfg: Mapping[str, Any] = DEFAULT_NORM_CFG,
        conv_cfg: Mapping[str, Any] = DEFAULT_CONV_CFG,
        init_cfg: Optional[Mapping[str, Any]] = None,
        residual: bool = False,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = nn.ModuleList()
        for i, layer_num in enumerate(layer_nums):
            block = nn.ModuleList()
            block.append(
                nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_filters[i],
                        out_channels[i],
                        3,
                        stride=layer_strides[i],
                        padding=1,
                    ),
                    build_norm_layer(norm_cfg, out_channels[i])[1],
                    nn.ReLU(inplace=True),
                )
            )
            for _ in range(layer_num):
                block.append(
                    nn.Sequential(
                        build_conv_layer(conv_cfg, out_channels[i], out_channels[i], 3, padding=1),
                        build_norm_layer(norm_cfg, out_channels[i])[1],
                        nn.ReLU(inplace=True),
                    )
                )

            blocks.append(block)

        self.blocks = blocks
        self.residual = residual

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            for _, conv in enumerate(block):
                temp = conv(x)
                if temp.shape == x.shape and self.residual:
                    x = temp + x
                else:
                    x = temp
            outs.append(x)
        return tuple(outs)
