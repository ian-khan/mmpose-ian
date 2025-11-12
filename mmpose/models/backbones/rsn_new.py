from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from mmengine.model import BaseModule

def is_power_of_2(number: int) -> bool:
    """Check whether `number` is power of 2.
    Args: `number`: the number to be checked.
    Returns: `bool`
    """
    assert number > 0, "Number must be positive."
    return number & (number - 1) == 0

# In a hierarchical neural network structure, the basic module on each level is represented by a class.
# Instances of a child module are created inside an instance of the parent module.

# Derived parameters of a module are those derived from the parameters its parent module.
# Their values vary across instances, and are derived during runtime.
# Therefore, they are set as positional parameters.

# Propagated parameters of a module are those copied from the corresponding parameters of its parent module.
# Their values are the same across instances, and do not have the derivation process.
# Therefore, they are set as keyword parameters with default values.

# A module also has the propagating parameter, which comes from its parent and goes to child.
# It adopts the form of a config dict, which nests the config dicts of the descendants.
# It is unpacked and passed to the constructor of its direct child.

class ConvStep(nn.Module):
    """A minimal Conv–Norm–Act step used in Residual Step Block (RSB).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Default: 3.
        stride (int): Convolution stride. Default: 1.
        padding (int): Convolution padding. Default: 1.
        norm_layer (nn.Module): Normalization layer class. Default: nn.BatchNorm2d.
        act_layer (nn.Module): Activation layer class. Default: nn.ReLU.
        inplace (bool): Whether to use inplace activation (if supported). Default: False.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 norm_layer= nn.BatchNorm2d,
                 act_layer= nn.ReLU,
                 inplace: bool = False):
        super().__init__()
        # when normalization is present, bias in convolution is redundant
        bias = norm_layer is None
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.norm = norm_layer(out_channels)
        # some activation function does not have the inplace argument
        try:
            self.act = act_layer(inplace=inplace)
        except TypeError:
            self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualStepsBlock(nn.Module):
    """
    Reimplementation of MMPose's Residual Step Block (RSB).

    When there is mismatch between the shapes of input and output feature maps,
    a convolution-normalization module is created on the skip connection to make them match.
    The final activation is applied after convergence of the 2 paths.

    Args:
        in_channels (int): Number of input channels. Derived.
        out_channels (int): Number of output channels. Derived.
        stride (int): The stride of the block. Derived.
        base_in_channels (int): base input channels of the first layer. Default: 64.
        base_branch_channels (int): base branch channels of the first layer. Default: 26.
        n_branches (int): the number of branches. Default: 4.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 base_in_channels: int=64,
                 base_branch_channels: int=26,
                 n_branches: int=4,
                 step_cfg: dict = None):
        super().__init__()

        div, mod = divmod(in_channels, base_in_channels)
        assert mod == 0, "in_channels should be some multiple of base_channel."
        assert n_branches >= 1, "There should be at least one branch."

        self.step_cfg = step_cfg or {}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_channels = div * base_branch_channels
        self.n_branches = n_branches
        self.stride = stride

        self.stem_diverge = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels = self.n_branches * self.branch_channels,
                      kernel_size=1,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(self.n_branches * self.branch_channels),
            nn.ReLU(inplace=False),
        )
        self.branched_steps = nn.ModuleList()
        for b in range(n_branches):
            self.branched_steps.append(nn.ModuleList())
            for s in range(self.n_branches):
                self.branched_steps[b].append(
                    ConvStep(in_channels=self.branch_channels,
                             out_channels=self.branch_channels,
                             **self.step_cfg)
                )
        self.stem_converge = nn.Sequential(
            nn.Conv2d(in_channels=self.n_branches * self.branch_channels,
                      out_channels=self.out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.skip_connection = None
        if self.in_channels != self.out_channels or self.stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
        self.converged_act = nn.ReLU(inplace=False)

    def forward(self, x):
        skip = x if self.skip_connection is None else self.skip_connection(x)

        x = self.stem_diverge(x)  # (B, n_b*C_b, H, W)
        x = torch.split(x, self.branch_channels, dim=1)  # n_b x (B, C_b, H, W)

        # The input and output tensors involved in the steps,
        # each tensor in the grid is of shape (B, C_b, H, W).
        tensor_grid = []
        for b in range(self.n_branches):
            tensor_grid.append([x[b]])
            for s in range(b + 1):
                input_tensor = tensor_grid[b][s]
                if s < b:
                    input_tensor = input_tensor + tensor_grid[b - 1][s + 1]
                output_tensor = self.branched_steps[b][s](input_tensor)
                tensor_grid[b].append(output_tensor)

        x = torch.cat([tensor_grid[b][b + 1] for b in range(self.n_branches)], dim=1)
        x = self.stem_converge(x)

        x = x + skip
        x = self.converged_act(x)
        return x


class DownsampleLayer(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 n_blocks: int,
                 block: nn.Module = ResidualStepsBlock,
                 block_cfg: dict = None):
        super().__init__()

        assert n_blocks >= 1, "There should be at least one block."

        self.block_cfg = block_cfg or {}

        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            # The first block in a layer is responsible for
            # channel expansion and down sampling.
            _in_channels = in_channels if i == 0 else out_channels
            _stride = stride if i == 0 else 1
            _block = block(in_channels=_in_channels,
                           out_channels=out_channels,
                           stride=_stride,
                           **self.block_cfg)
            self.blocks.append(_block)

    def forward(self, x):
        x = self.blocks(x)
        return x


class DownsampleModule(BaseModule):
    def __init__(self,
                 in_channels: int,
                 n_blocks: Sequence[int],
                 has_skip: bool = False,
                 layer_cfg: dict = None,):
        super().__init__()

        assert len(n_blocks) >= 1, "There should be at least one layer"
        self.n_layers = len(n_blocks)

        self.layer_cfg = layer_cfg or {}

        self.has_skip = has_skip

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            _in_channels = in_channels if i == 0 else in_channels * pow(2, i-1)
            _out_channels = in_channels if i == 0 else in_channels * pow(2, i)  # intentionally verbose
            _stride = 1 if i == 0 else 2
            _n_blocks = n_blocks[i]
            layer = DownsampleLayer(in_channels=_in_channels,
                                    out_channels=_out_channels,
                                    stride=_stride,
                                    n_blocks=_n_blocks,
                                    **self.layer_cfg)
            self.layers.append(layer)

    def forward(self,
                x: torch.Tensor,
                skip1: Optional[Sequence[torch.Tensor]] = None,
                skip2: Optional[Sequence[torch.Tensor]] = None) -> tuple[torch.Tensor]:
        downsample_out = list()
        for i in range(self.n_layers):
            x = self.layers[i](x)
            x = x if not self.has_skip else x + skip1[i] + skip2[i]
            downsample_out.append(x)
        downsample_out.reverse()
        return tuple(downsample_out)


class UpsampleLayer(BaseModule):
    def __init__(self,
                 dl_out_channels: int,
                 is_first_layer: bool,
                 has_cross_stage_skip: bool=False,
                 ul_out_channels: int=256,
                 ):
        super().__init__()

        self.is_first_layer = is_first_layer
        self.has_cross_stage_skip = has_cross_stage_skip

        # Not activated until converged
        self.dl_out_projection = nn.Sequential(
            nn.Conv2d(in_channels=dl_out_channels,
                      out_channels=ul_out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(ul_out_channels)
        )

        # Not activated until converged
        if not is_first_layer:
            self.ul_out_projection = nn.Sequential(
                nn.Conv2d(in_channels=ul_out_channels,
                          out_channels=ul_out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(ul_out_channels)
            )

        self.converged_act = nn.ReLU(inplace=False)

        # Skip connections are activated then added to dls' output
        if has_cross_stage_skip:
            self.dl_out_skip = nn.Sequential(
                nn.Conv2d(in_channels=dl_out_channels,
                          out_channels=dl_out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(dl_out_channels),
                nn.ReLU(inplace=False)
            )

            self.out_skip = nn.Sequential(
                nn.Conv2d(in_channels=ul_out_channels,
                          out_channels=dl_out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(dl_out_channels),
                nn.ReLU(inplace=False)
            )

    def forward(self, dl_out: torch.Tensor, ul_out: Optional[torch.Tensor]) -> tuple:
        """

        Args:
            dl_out: The feature map output from the down-sampling layer with the same output resolution.
            ul_out: The feature map output from the previous up-sampling layer.

        Returns: The output of this ul; Two skip connections to add to output of dl in the next stage of same out reso.

        """
        out = self.dl_out_projection(dl_out)
        if not self.is_first_layer:
            ul_out = F.interpolate(ul_out,
                                   scale_factor=2,
                                   mode='bilinear',
                                   align_corners=True)
            out = out + self.ul_out_projection(ul_out)
        out = self.converged_act(out)

        skip1 = self.dl_out_skip(dl_out) if self.has_cross_stage_skip else None
        skip2 = self.out_skip(out) if self.has_cross_stage_skip else None

        return out, skip1, skip2