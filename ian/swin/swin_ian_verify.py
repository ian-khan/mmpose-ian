import torch
from mmpose.models.backbones.rsn_ian_steps import SwinStep

step = SwinStep(in_channels=64,
                out_channels=64,
                win_size=(2, 2))
step_input = torch.randn(10 * 32 * 24, 4, 64)
step_output = step.forward(step_input)
print("\nSwin Step:")
print(" Input  shape:", step_input.shape)
print(" Output shape:", step_output.shape)