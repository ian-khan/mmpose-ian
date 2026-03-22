from pathlib import Path
from pose_estimation.analysis import Experiment

exps = [
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_ab-att-deep-0_td-hm_4xrsn50-c_2xb192-200e_coco-256x192"),
               exp_times="20260319_000337",
               label="Exp 0")
]

Experiment.draw_iter_lr_curve(exps, output="show")
Experiment.draw_epoch_avg_loss_curve(exps, output="show")
Experiment.draw_epoch_coco_ap_curve(exps, ylim=(0.0, 0.75), output="show")
Experiment.get_leader_board(exps, 5, 61, 5)