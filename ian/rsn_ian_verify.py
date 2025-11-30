# Sanity check for reimplemented RSN classes

from mmpose.models.backbones.rsn_ian import (Stem,
                                             ConvStep,
                                             ResidualStepsBlock,
                                             DownsampleLayer,
                                             UpsampleLayer,
                                             ResidualStepsNetworkStage,
                                             InterstageTransition,
                                             ResidualStepsNetwork)
import torch

cfg = dict(enable_stage_skip=False,)

stem = Stem(stage_in_channels=64)
stem_input = torch.randn(10, 3, 256, 192)
stem_output = stem(stem_input)
print("Stem:")
print(" Input  shape:", stem_input.shape)
print(" Output shape:", stem_output.shape)  # (10, 64, 64, 48)

step = ConvStep(26, 26)
step_input = torch.randn(10, 26, 64, 48)
step_output = step(step_input)
print("\nConv Step:")
print(" Input  shape:", step_input.shape)
print(" Output shape:", step_output.shape)  # (10, 26, 64, 48)

block = ResidualStepsBlock(64, 64, 1)
block_input = torch.randn(10, 64, 64, 48)
block_output = block(block_input)
print("\nResidual Step Block: (Layer 1 Block 1, stride=1)")
print(" Input  shape:", block_input.shape)
print(" Output shape:", block_output.shape)  # (10, 64, 64, 48)

dl = DownsampleLayer(in_channels=64,
                     out_channels=128,  # layer 1
                     stride=2,
                     n_blocks=2,
                     in_first_stage=False,
                     in_final_stage=False,)
dl_input = torch.randn((10, 64, 64, 48))
dl_skip = torch.randn((10, 128, 32, 24))
ul_skip = torch.randn((10, 128, 32, 24))
dl_output, dl_skip = dl(dl_input, dl_skip, ul_skip)
print("\nDownsample Layer #2:")
print(" Output shape:", dl_output.shape)  # (10, 128, 32, 24)
print(" Skip   shape:", dl_skip.shape)  # (10, 256, 32, 24)

ul = UpsampleLayer(dl_out_channels=128,
                   is_first_layer=False,
                   in_final_stage=False,)
ul_output = torch.randn((10, 256, 16, 12))
ul_output, ul_skip = ul(dl_output, ul_output)
print("\nUpsample Layer #3:")
print(" Output shape:", ul_output.shape)  # (10, 256, 32, 24)
print(" Skip   shape:", ul_skip.shape)  # (10, 128, 32, 24)

stage = ResidualStepsNetworkStage([2, 2, 6, 2],
                                  is_first_stage=False,
                                  is_final_stage=False,)
stage_input = torch.randn(10, 64, 64, 48)
factors = [pow(2, i) for i in range(4)]
dl_skips = [torch.randn((10, 64*f, 64//f, 48//f)) for f in factors]
ul_skips = [torch.randn((10, 64*f, 64//f, 48//f)) for f in factors]
stage_output, dl_skips, ul_skips = stage(stage_input, dl_skips, ul_skips)
print("\nStage output:")
for i in range(4):
    print(f" DL #{i} skip:", dl_skips[i].shape)
    print(f" UL #{i} output:", stage_output[i].shape)
    print(f" UL #{i} skip:", ul_skips[3-i].shape)

transition = InterstageTransition(stage_out_channels=256,
                                  stage_in_channels=64,)
transition_input = torch.randn(10, 256, 64, 48)
transition_output = transition(transition_input)
print("\nTransition:")
print(" Input  shape:", transition_input.shape)  # (10, 256, 64, 48)
print(" Output shape:", transition_output.shape)  # (10, 64, 64, 48)

rsn = ResidualStepsNetwork(stage_layer_blocks=[[2, 2, 6, 2],
                                               [2, 2, 6, 2],
                                               [2, 2, 6, 2],],
                           cfg=cfg,)
rsn_input = torch.randn(10, 3, 256, 192)
rsn_output = rsn(rsn_input)
print()
for i in range(3):
    print(f"RSN stage #{i} output:", *[rsn_output[i][j].shape for j in range(4)])  # all (10, 256, 64, 48)