import torch
from mmpose.models.backbones.rsn_swin import SwinStep

step = SwinStep(in_channels=26,
                input_size=(32, 24),
                shift=(0, 0))
step_input = torch.randn(10, 26, 32, 24)
step_output = step.forward(step_input)
print("\nSwin Step:")
print(" Input  shape:", step_input.shape)
print(" Output shape:", step_output.shape)

shifted_step = SwinStep(in_channels=26,
                        input_size=(32, 24),
                        shift=(1, 1))
shifted_step_input = torch.randn(10, 26, 32, 24)
shifted_step_output = shifted_step.forward(shifted_step_input)
print("\nShifted Swin Step:")
print(" Input  shape:", shifted_step_input.shape)
print(" Output shape:", shifted_step_output.shape)