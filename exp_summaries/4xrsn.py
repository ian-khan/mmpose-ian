"""Compare 4xRSN-Ian with RSB-C and 4xRSN-MMPose.
Both trained on 384x288 images for 210 epochs."""
from pose_estimation.analysis import Experiment
from pathlib import Path

exps = [
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_td-hm_4xrsn-alter-c_3xb64-210e_coco-384x288"),
               exp_times="20260112_154914",
               label="4xRSN-Ian RSB-C"),
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_td-hm_4xrsn_3xb64-210e_coco-384x288"),
               exp_times=["20260403_175449"],
               label="4xRSN-MMPose"),
]

Experiment.draw_iter_lr_curve(exps, output="show")
Experiment.draw_epoch_avg_loss_curve(exps, output="show")
Experiment.draw_epoch_coco_ap_curve(exps, output="show")
# Experiment.get_leader_board(exps, 5, 201, 5)