from pose_estimation.analysis import Experiment
from pathlib import Path

exps = [Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/"
                                 "work_dirs/ian_td-hm_rsn-ian_8xb32-210e_coco-256x192"),
                   exp_times=["20251126_173025"], label="M-Style 210E"),
        Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                 "ian_td-hm_rsn-ian_3xb128-360e_coco-256x192_it-ol"),
                   exp_times=["20260121_214515",
                              "20260122_200638",
                              "20260123_124145",
                              "20260123_174904",
                              "20260124_000245"], label="R-Style 360E")]

Experiment.draw_iter_lr_curve(exps, output="show")
Experiment.draw_epoch_avg_loss_curve(exps, output="show")
Experiment.draw_epoch_coco_ap_curve(exps, output="show")