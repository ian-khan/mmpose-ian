# Print the named parameters of reimplemented RSN
# to figure out what parameters are not used for loss calculation
from mmpose.models.backbones.rsn_new import ResidualStepsNetwork

rsn = ResidualStepsNetwork([[3, 4, 6, 3]])

for idx, (name, p) in enumerate(rsn.named_parameters()):
    print(idx, name, p.shape)