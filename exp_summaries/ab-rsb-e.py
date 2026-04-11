from pathlib import Path
from pose_estimation.analysis import Experiment

experiments = [
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_ab-blk-att-2_td-hm_rsn50-c_3xb128-200e_coco-256x192"),
               exp_times="20260209_133443",
               label="RSB-C Exp 2"),
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_ab-rsb-e-0_td-hm_rsn50_3xb128-200e_coco-256x192"),
               exp_times="20260407_152714",
               label="RSB-E Exp 0"),
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_ab-rsb-e-1_td-hm_rsn50_3xb128-200e_coco-256x192"),
               exp_times=["20260408_211759", "20260409_174028"],
               label="RSB-E Exp 1"),
Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_ab-rsb-e-2_td-hm_rsn50_3xb128-200e_coco-256x192"),
               exp_times=["20260410_193705", "20260411_005043"],
               label="RSB-E Exp 2"),
]

Experiment.draw_iter_lr_curve(experiments, output="show")
Experiment.draw_epoch_avg_loss_curve(experiments, output="show")
Experiment.draw_epoch_coco_ap_curve(experiments, xlim=(0.0, 200.0), ylim=(0.4, 0.74), output="show")
# Experiment.get_leader_board(experiments, 5, 201, 5)