import torch
from torch import nn
from typing import Optional

class SwinStep(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 head_channels: int=32,
                 win_size: tuple[int, int]=(2, 2),
                 qkv_bias: bool=True,
                 qk_scale: Optional[float]=None,
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 **kwargs):
        super().__init__()

        assert out_channels % head_channels == 0, \
            "out_channels must be divisible by head_channels"

        self.head_channels = head_channels
        self.num_heads = out_channels // head_channels
        self.win_size = win_size
        self.qk_scale = qk_scale or head_channels ** -0.5

        self.bias_bucket = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), self.num_heads))
        # bias bucket initialization
        rel_pos_to_bias_bucket = \
            self.compute_rel_pos_to_bias_bucket(win_size[0], win_size[1])
        self.register_buffer('rel_pos_to_bias_bucket', rel_pos_to_bias_bucket)

        self.qkv = nn.Linear(in_channels,
                             out_channels * 3,
                             bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(out_channels,
                              out_channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        B: batch size;
        W: number of windows in the feature map;
        T: number of tokens in a window, equal to wh * ww;
        C: number of channels in the feature map;
        Args:
            x: input tensor
            mask: mask tensor of shape (W, T, T)

        Returns:

        """
        BxW, T, C = x.shape
        assert T == self.win_size[0] * self.win_size[1], "Number of tokens per window must be equal to wh * ww"

        qkv = self.qkv(x).reshape(BxW, T, 3, self.num_heads, self.head_channels)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, BxW, nh, T, Ch
        q, k, v = qkv[0], qkv[1], qkv[2]  # BxW, nh, T, Ch

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale  # BxW, nh, T, T

        bias = self.bias_bucket[self.rel_pos_to_bias_bucket.view(-1)].view(T, T, self.num_heads)  # T, T, nh
        bias = bias.permute(2, 0, 1).contiguous()  # nh, T, T
        attn = attn + bias  # BxW, nh, T, T

        if mask is not None:
            W = mask.shape[0]
            attn = attn.view(BxW // W, W, self.num_heads, T, T)  # B, W, nh, T, T
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(BxW, self.num_heads, T, T)  # BxW, nh, T, T

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(BxW, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def compute_rel_pos_to_bias_bucket(win_h: int,
                                       win_w: int) -> torch.Tensor:
        """
        Compute the mapping from relative position
        between query token and key token to bias bucket.
        Args:
            win_h: height of the window
            win_w: width of the window

        Returns: the mapping

        """
        # Create the coordinate sequence for tokens in a window
        coord_h = torch.arange(win_h)
        coord_w = torch.arange(win_w)
        coord_seq = torch.cartesian_prod(coord_h, coord_w)  # wh*ww, 2

        # Calculate pairwise relative positions between query and key tokens
        # Each row uses the same query token and different key tokens
        rel_pos = coord_seq[:, None, :] - coord_seq[None, :, :]  # wh*ww, wh*ww, 2

        # Range of relative position is [-(wh-1), wh-1], [-(ww-1), ww-1]
        # Shift this range to non-negative [0, 2*(wh-1)], [0, 2*(ww-1)]
        rel_pos[:, :, 0] += win_h - 1
        rel_pos[:, :, 1] += win_w - 1

        # Map relative position to a number
        rel_pos[:, :, 0] *= 2 * win_w - 1
        rel_pos_to_bias_bucket = rel_pos.sum(-1)
        return rel_pos_to_bias_bucket
