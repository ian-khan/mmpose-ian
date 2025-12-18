from mmpose.models.backbones.rsn import RSB
from mmpose.models.backbones.rsn_ian import ResidualStepsBlock
from mmpose.models.backbones.rsn_swin import ResidualSwinStepsBlock

from calflops import calculate_flops

rsb_swin_cfg = dict({"win_size": (2, 2)})

rsb = RSB(in_channels=64, out_channels=64)
rsb_ian = ResidualStepsBlock(in_channels=64, out_channels=64, stride=1)
rsb_swin = ResidualSwinStepsBlock(in_channels=64, out_channels=64, stride=1, cfg=rsb_swin_cfg)

input_shape = (1, 64, 64, 48)

for model in (rsb, rsb_ian, rsb_swin):
    flops, macs, params = calculate_flops(model,
                                          input_shape,
                                          print_results=False,
                                          output_as_string=True,
                                          output_precision=4)
    print(f"{str(model.__class__.__name__).ljust(22)}: "
          f"FLOPs: {flops}, MACs: {macs}, Params: {params}")