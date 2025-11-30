from mmpose.registry import MODELS

import copy
from typing import Sequence  # collections.abc.Sequence unavailable for Python <= 3.9
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from mmengine.model import BaseModule

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

class Stem(nn.Module):
    def __init__(self,
                 stage_in_channels: int=64,
                 **kwargs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=stage_in_channels,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(stage_in_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x


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
                 inplace: bool = False,
                 **kwargs):
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
        stage_in_channels (int): base input channels of the first layer. Default: 64.
        base_branch_channels (int): base branch channels of the first layer. Default: 26.
        n_branches (int): the number of branches. Default: 4.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 stage_in_channels: int=64,
                 base_branch_channels: int=26,
                 n_branches: int=4,
                 cfg: dict = None,
                 **kwargs):
        super().__init__()

        div, mod = divmod(in_channels, stage_in_channels)
        assert mod == 0, "in_channels should be some multiple of base_channel."
        assert n_branches >= 1, "There should be at least one branch."

        cfg = cfg or {}

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
            # the b-th branch has b steps
            for s in range(b+1):
                self.branched_steps[b].append(
                    ConvStep(in_channels=self.branch_channels,
                             out_channels=self.branch_channels,
                             **cfg)
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
                 in_first_stage: bool,
                 in_final_stage: bool,
                 block: nn.Module = ResidualStepsBlock,
                 enable_stage_skip: bool = True,  # False for debug; True for deploy
                 cfg: dict = None,
                 **kwargs):
        super().__init__()

        assert n_blocks >= 1, "There should be at least one block."

        cfg = cfg or {}

        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            # The first block in a layer is responsible for
            # channel expansion and down sampling.
            _in_channels = in_channels if i == 0 else out_channels
            _stride = stride if i == 0 else 1
            _block = block(in_channels=_in_channels,
                           out_channels=out_channels,
                           stride=_stride,
                           **cfg)
            self.blocks.append(_block)

        self.receive_in_skip = enable_stage_skip and not in_first_stage
        self.compute_out_skip = enable_stage_skip and not in_final_stage
        if self.compute_out_skip:
            self.dl_skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            )

    def forward(self,
                dl_out: torch.Tensor,
                dl_skip: torch.Tensor = None,
                ul_skip: torch.Tensor = None, ) -> tuple:
        """
        Forward pass. Tensors not prefixed with underscores are inputs;
        tensors prefixed with underscores are outputs;

        Args:
            dl_out: Feature map from the previous down-sampling layer.
            dl_skip: Feature map from down-sampling layer in the previous stage.
            ul_skip: Feature map from up-sampling layer in the previous stage.

        Returns: Output and skip connection from this down-sampling layer.

        """
        _dl_out = self.blocks(dl_out)
        if self.receive_in_skip:
            _dl_out = _dl_out + dl_skip + ul_skip

        _dl_skip = self.dl_skip_connection(_dl_out) if self.compute_out_skip else None

        return _dl_out, _dl_skip


class UpsampleLayer(BaseModule):
    def __init__(self,
                 dl_out_channels: int,
                 is_first_layer: bool,
                 in_final_stage: bool,
                 ul_out_channels: int = 256,
                 enable_stage_skip: bool=True,  # False for debug; True for deploy
                 **kwargs):
        super().__init__()

        self.is_first_layer = is_first_layer
        self.compute_out_skip = enable_stage_skip and not in_final_stage

        # Not activated until fused
        self.dl_projection = nn.Sequential(
            nn.Conv2d(in_channels=dl_out_channels,
                      out_channels=ul_out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(ul_out_channels)
        )

        # Not activated until fused
        if not is_first_layer:
            self.ul_projection = nn.Sequential(
                nn.Conv2d(in_channels=ul_out_channels,
                          out_channels=ul_out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(ul_out_channels)
            )

        self.fused_act = nn.ReLU(inplace=False)

        # Skip connections are activated then added to DLs' output
        if self.compute_out_skip:
            self.ul_skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=ul_out_channels,
                          out_channels=dl_out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(dl_out_channels),
                nn.ReLU(inplace=False)
            )

    def forward(self,
                dl_out: torch.Tensor,
                ul_out: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass. Tensors not prefixed with underscores are inputs;
        tensors prefixed with underscores are outputs;

        Args:
            dl_out: Feature map from the down-sampling layer on same level in this stage.
            ul_out: Feature map from the previous up-sampling layer.

        Returns: Output and skip connection from this up-sampling layer.

        """
        _ul_out = self.dl_projection(dl_out)
        if not self.is_first_layer:
            ul_out = F.interpolate(ul_out,
                                   scale_factor=2,
                                   mode='bilinear',
                                   align_corners=True)
            _ul_out = _ul_out + self.ul_projection(ul_out)
        _ul_out = self.fused_act(_ul_out)

        _ul_skip = self.ul_skip_connection(_ul_out) if self.compute_out_skip else None

        return _ul_out, _ul_skip


class ResidualStepsNetworkStage(nn.Module):
    def __init__(self,
                 n_blocks: Sequence[int],
                 is_first_stage: bool,
                 is_final_stage: bool,
                 enable_layer_supervision: bool = True,
                 stage_in_channels: int=64,
                 cfg: dict = None,
                 **kwargs):
        super().__init__()

        self.n_levels = len(n_blocks)
        assert self.n_levels > 0, "There must be at least one level of resolution."

        cfg = cfg or {}

        self.is_first_stage = is_first_stage
        self.enable_layer_supervision = enable_layer_supervision

        self.down_layers = nn.ModuleList()
        self.up_layers = list()
        for i in range(self.n_levels):
            dl_in_channels = stage_in_channels if i == 0 else stage_in_channels * pow(2, i - 1)
            dl_out_channels = stage_in_channels if i == 0 else stage_in_channels * pow(2, i)
            stride = 1 if i == 0 else 2
            is_first_layer = i == self.n_levels - 1  # self.up_layers is later reversed
            self.down_layers.append(DownsampleLayer(in_channels=dl_in_channels,
                                                    out_channels=dl_out_channels,
                                                    stride=stride,
                                                    n_blocks=n_blocks[i],
                                                    in_first_stage=is_first_stage,
                                                    in_final_stage=is_final_stage,
                                                    cfg=cfg,
                                                    **cfg))
            self.up_layers.append(UpsampleLayer(dl_out_channels=dl_out_channels,
                                                is_first_layer=is_first_layer,
                                                in_final_stage=is_final_stage,
                                                **cfg))
        self.up_layers.reverse()
        self.up_layers = nn.ModuleList(self.up_layers)

    def forward(self,
                x: torch.Tensor,
                dl_skips: Optional[Sequence[torch.Tensor]] = None,
                ul_skips: Optional[Sequence[torch.Tensor]] = None) -> tuple:
        assert not ((dl_skips is None) ^ (ul_skips is None)), \
            ("Skip connections from the down-sampling layers and the up-sampling layers"
             "should be given at the same time.")

        _dl_out = x
        _dl_outs = list()
        _dl_skips = list()
        _ul_out = None
        _ul_outs = list() if self.enable_layer_supervision else None
        _ul_skips = list()

        for i in range(self.n_levels):
            _dl_out, _dl_skip = self.down_layers[i](dl_out=_dl_out,
                                                    dl_skip=None if self.is_first_stage else dl_skips[i],
                                                    ul_skip=None if self.is_first_stage else ul_skips[i])
            _dl_outs.append(_dl_out)
            _dl_skips.append(_dl_skip)
        _dl_outs.reverse()

        for i in range(self.n_levels):
            _ul_out, _ul_skip = self.up_layers[i](dl_out=_dl_outs[i],
                                                  ul_out=_ul_out)
            if self.enable_layer_supervision:
                _ul_outs.append(_ul_out)
            _ul_skips.append(_ul_skip)
        _ul_skips.reverse()

        return (tuple(_ul_outs) if self.enable_layer_supervision else _ul_out,
                _dl_skips, _ul_skips)


class InterstageTransition(nn.Module):
    def __init__(self,
                 stage_out_channels: int=256,
                 stage_in_channels: int=64,
                 **kwargs):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=stage_out_channels,
                      out_channels=stage_in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(stage_in_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transition(x)
        return x


@MODELS.register_module()
class ResidualStepsNetwork(nn.Module):
    def __init__(self,
                 stage_layer_blocks: Sequence[Sequence[int]],
                 enable_layer_supervision: bool = True,
                 cfg: dict = None,
                 **kwargs):
        super().__init__()

        cfg = copy.deepcopy(cfg) or {}

        self.n_stages = len(stage_layer_blocks)
        assert self.n_stages > 0, "There must be at least one stage."

        self.enable_layer_supervision = enable_layer_supervision

        self.stem = Stem(**cfg)

        self.stages = nn.ModuleList()
        for i in range(self.n_stages):
            is_first_stage = i == 0
            is_final_stage = i == self.n_stages - 1
            stage = ResidualStepsNetworkStage(n_blocks=stage_layer_blocks[i],
                                              is_first_stage=is_first_stage,
                                              is_final_stage=is_final_stage,
                                              cfg=cfg,
                                              **cfg)
            self.stages.append(stage)

        self.transitions = nn.ModuleList()
        for i in range(self.n_stages-1):
            transition = InterstageTransition(**cfg)
            self.transitions.append(transition)


    def forward(self, x: torch.Tensor) -> tuple:
        x = self.stem(x)

        dl_skips = None
        ul_skips = None
        stage_outs = list()
        for i in range(self.n_stages):
            ul_output, dl_skips, ul_skips = self.stages[i](x, dl_skips, ul_skips)
            stage_outs.append(ul_output)
            x = ul_output[-1] if self.enable_layer_supervision else ul_output
            if i < self.n_stages - 1:
                x = self.transitions[i](x)

        return tuple(stage_outs)

