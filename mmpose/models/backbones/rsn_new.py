import torch
from torch import nn
from mmengine.model import BaseModule

def is_power_of_2(number: int) -> bool:
    """Check whether `number` is power of 2.
    Args: `number`: the number to be checked.
    Returns: `bool`
    """
    assert number > 0, "Number must be positive."
    return number & (number - 1) == 0


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
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_channels: int=64,
                 base_branch_channels: int=26,
                 n_branches: int=4,
                 stride: int=1,):
        super().__init__()

        div, mod = divmod(in_channels, base_channels)
        assert mod == 0, "in_channels should be some multiple of base_channel."
        assert n_branches >= 1, "There should be at least one branch."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_channels = div * base_branch_channels
        self.n_branches = n_branches
        self.stride = stride

        self.stem_diverge = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels = self.n_branches * self.branch_channels,
                      kernel_size=1,
                      stride=stride,),
            nn.BatchNorm2d(self.n_branches * self.branch_channels),
            nn.ReLU(inplace=False),
        )
        self.branched_steps = nn.ModuleList()
        for b in range(n_branches):
            self.branched_steps.append(nn.ModuleList())
            for s in range(self.n_branches):
                self.branched_steps[b].append(
                    ConvStep(in_channels=self.branch_channels,
                             out_channels=self.branch_channels,)
                )
        self.stem_converge = nn.Sequential(
            nn.Conv2d(in_channels=self.n_branches * self.branch_channels,
                      out_channels=self.out_channels,
                      kernel_size=1,),
            nn.BatchNorm2d(self.out_channels),
        )
        self.skip_connection = None
        if self.in_channels != self.out_channels or self.stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          kernel_size=1,
                          stride=stride),
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
                 block: nn.Module,
                 n_blocks: int,
                 in_channels: int,
                 out_channels: int,
                 base_channels: int = 64,
                 base_branch_channels: int = 26,
                 n_branches: int = 4,
                 stride: int = 1,):
        super().__init__()

        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            # The first block in a layer is responsible for
            # channel expansion and down sampling.
            _in_channels = in_channels if i == 0 else out_channels
            _stride = stride if i == 0 else 1
            _block = block(in_channels=_in_channels,
                           out_channels=out_channels,
                           base_channels=base_channels,
                           base_branch_channels=base_branch_channels,
                           n_branches=n_branches,
                           stride=_stride,)
            self.blocks.append(_block)

    def forward(self, x):
        x = self.blocks(x)
        return x
