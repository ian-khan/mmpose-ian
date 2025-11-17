# Sanity check for reimplemented RSN classes

from mmpose.models.backbones.rsn_new import (Stem,
                                             ConvStep,
                                             ResidualStepsBlock,
                                             DownsampleLayer,
                                             UpsampleLayer,
                                             ResidualStepsNetworkStage,
                                             InterstageTransition,
                                             ResidualStepsNetwork)
import torch

stem = Stem(stage_in_channels=64)
stem_input = torch.randn(10, 3, 256, 192)
stem_output = stem(stem_input)
print("Stem output: ", stem_output.shape)  # (10, 64, 64, 48)

dl = DownsampleLayer(in_channels=64,
                     out_channels=128,  # layer 1
                     stride=2,
                     n_blocks=2,
                     in_first_stage=False,
                     in_final_stage=False)
dl_input = torch.randn((10, 64, 64, 48))
dl_output, dl_skip = dl(dl_input)
print("\nDL #1 output: ", dl_output.shape)  # (10, 128, 32, 24)
print("DL skip: ", dl_skip)

ul = UpsampleLayer(dl_out_channels=128,
                   is_first_layer=False,
                   in_final_stage=False)
ul_output = torch.randn((10, 256, 16, 12))
ul_output, ul_skip = ul(dl_output, ul_output)
print("\nUL #2 output: ", ul_output.shape)  # (10, 256, 32, 24)
print("UL skip: ", ul_skip)

cfg = dict(
    enable_stage_skip=True,
)
stage = ResidualStepsNetworkStage([2, 2, 6, 2],
                                  is_first_stage=False,
                                  is_final_stage=False,
                                  cfg=cfg,
                                  **cfg)
stage_input = torch.randn(10, 64, 64, 48)
factors = [pow(2, i) for i in range(4)]
dl_skips = [torch.randn((10, 64*f, 64//f, 48//f)) for f in factors]
ul_skips = [torch.randn((10, 64*f, 64//f, 48//f)) for f in factors]
stage_output, dl_skips, ul_skips = stage(stage_input, dl_skips, ul_skips)
print("\nStage output: ", stage_output.shape)  # (10, 256, 64, 48)
for i in range(4):
    print(f" DL #{i} skip: ", dl_skips[i].shape)
    print(f" UL #{3-i} skip: ", ul_skips[i].shape)

transition = InterstageTransition(stage_out_channels=256,
                                  stage_in_channels=64,)
transition_input = torch.randn(10, 256, 64, 48)
transition_output = transition(transition_input)
print("\nTransition output: ", transition_output.shape)  # (10, 64, 64, 48)

rsn = ResidualStepsNetwork(stage_layer_blocks=[[2, 2, 6, 2],
                                               [2, 2, 6, 2],
                                               [2, 2, 6, 2],],
                           cfg=cfg,)
rsn_input = torch.randn(10, 3, 256, 192)
rsn_output = rsn(rsn_input)
print("\n")
for i in range(3):
    print(f"RSN stage #{i} output: ", rsn_output[i].shape)  # all (10, 256, 64, 48)