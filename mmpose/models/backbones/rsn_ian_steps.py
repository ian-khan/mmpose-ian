import torch
from torch import nn
from typing import Optional

class SwinStep(nn.Module):
    """
        B: batch size;
        mh, mw: height and width of the map;
        wh, ww: height and width of the window;
        W: number of windows in the feature map;
        H: number of heads;
        T: number of tokens in a window, equal to wh * ww;
        C: number of channels in the feature map;
        hc: number of channels per head;
    """
    def __init__(self,
                 in_channels: int,
                 input_size: tuple[int, int],
                 shift: tuple[int, int],
                 head_channels: int=32,
                 mlp_ratio: int=4,
                 win_size: tuple[int, int]=(2, 2),
                 qkv_bias: bool=True,
                 qk_scale: Optional[float]=None,
                 attn_drop: float=0.,
                 linear_drop: float=0.,
                 act_layer: nn.Module=nn.GELU,
                 norm_layer: nn.Module=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        assert in_channels % head_channels == 0, \
            "in_channels must be divisible by head_channels"
        assert input_size[0] % win_size[0] == 0 and input_size[1] % win_size[1] == 0,\
            "input_size must be divisible by win_size"
        assert 0 <= shift[0] < win_size[0] and 0 <= shift[1] < win_size[1], \
            "shift must be in [0, win_size)"

        self.head_channels = head_channels
        self.num_heads = in_channels // head_channels
        self.input_size = input_size
        self.win_size = win_size
        self.qk_scale = qk_scale or head_channels ** -0.5
        self.shift = shift
        self.do_shift = min(shift) > 0

        self.norm1 = norm_layer(in_channels)

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(linear_drop)

        self.norm2 = norm_layer(in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * mlp_ratio),
            act_layer(),
            nn.Dropout(linear_drop),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(linear_drop),
        )

        # Relative position biases
        num_rel_pos = (2 * win_size[0] - 1) * (2 * win_size[1] - 1)
        self.bias_buckets = nn.Parameter(torch.zeros(num_rel_pos, self.num_heads))
        # TODO: bias buckets initialization

        # Relative position biases index
        rel_pos_to_bias_bucket = self.compute_rel_pos_to_bias_bucket(win_size)
        self.register_buffer('rel_pos_to_bias_bucket', rel_pos_to_bias_bucket)

        # Attention mask, only needed for Shifted Window Multihead Self-Attention
        if self.do_shift:
            attn_mask = self.compute_attn_mask(input_size, win_size, shift)
            self.register_buffer('attn_mask', attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get and compute the tensor shapes-related variables
        B, mh, mw, C = x.shape
        assert (mh, mw) == self.input_size, "Input resolution is not correct"
        wh, ww = self.win_size
        W = (mh // wh) * (mw // ww)
        T = wh * ww

        shortcut = x

        x = self.norm1(x)  # B, mh, mw, C

        # Cyclic shift
        if self.do_shift:
            x = torch.roll(x, shifts=(-self.shift[0], -self.shift[1]), dims=(1, 2))

        # Partition windows
        x = self.window_partition(x, self.win_size)  # BxW, wh, ww, C

        # Compute q, k, v
        qkv = self.qkv(x)  # BxW, wh, ww, 3C
        # Split heads and flatten window tokens
        qkv = qkv.reshape(B*W, T, 3, self.num_heads, self.head_channels)  # BxW, T, 3, H, hc
        # Get ready for per-head matrix multiplication
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # 3, BxW, H, T, hc
        # Split q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # BxW, H, T, hc

        # Compute normalized attention scores
        attn = (q @ k.transpose(-2, -1)) * self.qk_scale  # BxW, H, T, T

        # Add the relative position biases
        bias = self.bias_buckets[self.rel_pos_to_bias_bucket.view(-1)]  # T*T, H
        bias = bias.view(T, T, self.num_heads)  # T, T, H
        bias = bias.permute(2, 0, 1).contiguous()  # H, T, T
        attn = attn + bias  # BxW, H, T, T

        # Apply the attention mask
        if self.do_shift:
            attn = attn.view(B, W, self.num_heads, T, T)  # B, W, H, T, T
            attn = attn + self.attn_mask  # mask: 1, W, 1, T, T
            attn = attn.view(B*W, self.num_heads, T, T)  # BxW, H, T, T

        # Softmax then dropout
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Attention output
        x = attn @ v  # BxW, H, T, hc
        # Concatenate heads and unflatten window
        x = x.transpose(1, 2).reshape(B*W, wh, ww, C)  # BxW, wh, ww, C

        # Projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        # Stitch windows
        x = self.window_stitch(x, self.input_size)  # B, mh, mw, C

        # Reverse cyclic shift
        if self.do_shift:
            x = torch.roll(x, shifts=self.shift, dims=(1, 2))

        x = x + shortcut  # B, mh, mw, C

        # MLP
        x = self.mlp(self.norm2(x)) + x  # B, mh, mw, C
        return x

    @staticmethod
    def compute_rel_pos_to_bias_bucket(win_size: tuple[int, int]) -> torch.Tensor:
        """
        Compute the mapping from relative position between query token and key token to bias bucket,
        which is the same for each window, and different for each head.
        Args:
            win_size: Spatial size of the window

        Returns: the mapping (wh*ww, wh*ww) or (T, T)

        """
        wh, ww = win_size

        # Create the coordinate sequence for tokens in a window
        coord_h = torch.arange(wh)
        coord_w = torch.arange(ww)
        coord_seq = torch.cartesian_prod(coord_h, coord_w)  # wh*ww, 2

        # Calculate pairwise relative positions between query and key tokens
        # Each row uses the same query token and different key tokens
        rel_pos = coord_seq[:, None, :] - coord_seq[None, :, :]  # wh*ww, wh*ww, 2

        # Range of relative position is [-(wh-1), wh-1], [-(ww-1), ww-1]
        # Shift this range to non-negative [0, 2*(wh-1)], [0, 2*(ww-1)]
        rel_pos[:, :, 0] += wh - 1
        rel_pos[:, :, 1] += ww - 1

        # Map relative position to a number
        rel_pos[:, :, 0] *= 2 * ww - 1
        rel_pos_to_bias_bucket = rel_pos.sum(-1)
        return rel_pos_to_bias_bucket

    @staticmethod
    def compute_attn_mask(input_size, win_size, shift) -> torch.Tensor:
        """
        Compute the attention mask, which is different for each window and
        same for each head.
        Args:
            input_size: Spatial size of the input feature map
            win_size: Spatial size of the window
            shift: The shift for window partition

        Returns: The attention mask (1, W, 1, T, T)

        """
        region_chart = torch.zeros((1, input_size[0], input_size[1], 1))
        h_slices = (slice(0, -win_size[0]),
                    slice(-win_size[0], -shift[0]),
                    slice(-shift[0], None))
        w_slices = (slice(0, -win_size[1]),
                    slice(-win_size[1], -shift[1]),
                    slice(-shift[1], None))
        region_index = 0
        for h in h_slices:
            for w in w_slices:
                region_chart[0, h, w, 0] = region_index
                region_index += 1

        region_chart = SwinStep.window_partition(region_chart, win_size)  # W, wh, ww, 1
        region_chart = region_chart.reshape(-1, win_size[0]*win_size[1])  # W, wh*ww or W, T

        attn_mask = region_chart[:, :, None] - region_chart[:, None, :]  # W, T, T
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0).masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(0)  # 1, W, 1, T, T
        return attn_mask

    @staticmethod
    def window_partition(x: torch.Tensor, win_size) -> torch.Tensor:
        B, mh, mw, C = x.shape
        wh, ww = win_size
        x = x.view(B, mh // wh, wh, mw // ww, ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh, ww, C)  # BxW, wh, ww, C
        return x

    @staticmethod
    def window_stitch(x: torch.Tensor, map_size) -> torch.Tensor:
        BxW, wh, ww, C = x.shape
        mh, mw = map_size
        W = (mh // wh) * (mw // ww)
        B = BxW // W
        x = x.view(B, mh // wh, mw // ww, wh, ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, mh, mw, C)
        return x

