from mmpose.models.backbones.rsn_new import (ConvStep,
                                             ResidualStepsBlock,
                                             DownsampleLayer, UpsampleLayer,
                                             ResidualStepsNetworkStage,
                                             Stem, InterstageTransition)
import torch

stem = Stem(stage_in_channels=64)
stem_input = torch.randn(10, 3, 256, 192)
stem_output = stem(stem_input)
print(stem_output.shape)

transition = InterstageTransition(stage_out_channels=256,
                                  stage_in_channels=64,)
transition_input = torch.randn(10, 256, 64, 48)
transition_output = transition(transition_input)
print(transition_output.shape)