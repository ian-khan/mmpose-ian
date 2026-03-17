from pose_estimation.analysis import Experiment
from pathlib import Path

exps = [Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                 "ian_td-hm_4xrsn-alter-c_3xb64-210e_coco-384x288"),
                   exp_times="20260112_154914",
                   label="4xRSN-Ian 210e"),
        Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                 "ian_test_td-hm_hrnet-w32_3xb48-200e_coco-384x288"),
                   exp_times=["20260313_191326"],
                   label="HRNet-Ian Initial Test")]

Experiment.draw_iter_lr_curve(exps)
Experiment.draw_epoch_avg_loss_curve(exps)
Experiment.draw_epoch_coco_ap_curve(exps)