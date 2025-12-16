import torch
from mmpose.models.backbones.rsn_swin import SwinStep, ResidualSwinStepsBlock
from mmpose.models.backbones.rsn_ian import ResidualStepsNetwork

step = SwinStep(in_channels=26,
                input_size=(32, 24),
                shift=(0, 0))
step_input = torch.randn(10, 32, 24, 26)
step_output = step.forward(step_input)
print("\nSwin Step:")
print(" Input  shape:", step_input.shape)
print(" Output shape:", step_output.shape)

shifted_step = SwinStep(in_channels=26,
                        input_size=(32, 24),
                        shift=(1, 1))
shifted_step_input = torch.randn(10, 32, 24, 26)
shifted_step_output = shifted_step.forward(shifted_step_input)
print("\nShifted Swin Step:")
print(" Input  shape:", shifted_step_input.shape)
print(" Output shape:", shifted_step_output.shape)

layer1_block = ResidualSwinStepsBlock(in_channels=64,
                                      out_channels=64,
                                      stride=1,)
layer1_block_input = torch.randn(10, 64, 64, 48)
layer1_block_output = layer1_block.forward(layer1_block_input)
print("\nLayer 1 Block:")
print(" Input  shape:", layer1_block_input.shape)
print(" Output shape:", layer1_block_output.shape)

layer2_block1 = ResidualSwinStepsBlock(in_channels=64,
                                       out_channels=128,
                                       stride=2,)
layer2_block1_input = torch.randn(10, 64, 64, 48)
layer2_block1_output = layer2_block1.forward(layer2_block1_input)
print("\nLayer 2 Block 1:")
print(" Input  shape:", layer2_block1_input.shape)
print(" Output shape:", layer2_block1_output.shape)

cfg = dict({"block": ResidualSwinStepsBlock})
stage_layer_blocks = [[2, 2, 2, 2],]
rsn = ResidualStepsNetwork(stage_layer_blocks,
                           cfg=cfg,
                           **cfg)
rsn_input = torch.randn(10, 3, 256, 192)
rsn_output = rsn(rsn_input)
num_stages = len(stage_layer_blocks)
num_layers = len(stage_layer_blocks[0])
print()
for i in range(num_stages):
    print(f"RSN stage #{i} output:", *[rsn_output[i][j].shape for j in range(num_layers)])
