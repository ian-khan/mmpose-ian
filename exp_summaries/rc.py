from pose_estimation.analysis import Experiment
from pathlib import Path

exps = [
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_td-hm_4xrsn-alter-c_3xb64-210e_coco-384x288"),
               exp_times="20260112_154914",
               label="Exp 210e"),
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_rc_td-hm_4xrsn50-c_3xb64-360e_coco-384x288"),
               exp_times=["20260215_165825", "20260219_123923",
                          "20260226_131658", "20260305_173510",
                          "20260306_222237", "20260308_001251",
                          "20260311_221326", "20260312_174947"],
               label="Exp -1"),
    Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                             "ian_rc-0_td-hm_4xrsn50-c_3x64-360e_coco-384x288"),
               exp_times=["20260322_115737",],
               label="Exp 0"),
]

Experiment.draw_iter_lr_curve(exps, output="show")
Experiment.draw_epoch_avg_loss_curve(exps, output="show")
Experiment.draw_epoch_coco_ap_curve(exps, output="show")