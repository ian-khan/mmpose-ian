from pathlib import Path
from pose_estimation.analysis import Experiment

experiments = [Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-0_td-hm_rsn50-d_3xb128-200e_coco-256x192"),
                          exp_times="20260206_193130",
                          label="Exp 0"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-att-1_td-hm_rsn50-c_3xb128-200e_coco-256x192"),
                          exp_times="20260208_043650",
                          label="Exp 1 (Baseline)"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-att-2_td-hm_rsn50-c_3xb128-200e_coco-256x192"),
                          exp_times="20260209_133443",
                          label="Exp 2"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-att-3_td-hm_rsn50-c_3xb128-200e_coco-256x192"),
                          exp_times="20260210_222517",
                          label="Exp 3"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-att-4_td-hm_rsn50-c_3xb128-200e_coco-256x192"),
                          exp_times=["20260212_073359",
                                     "20260213_225821"],
                          label="Exp 4"),
               ]

Experiment.draw_epoch_coco_ap_curve(experiments, ylim=(0.65, 0.75), output="show")
Experiment.draw_epoch_avg_loss_curve(experiments, ylim=(80.0, 100.0), output="show")
Experiment.draw_iter_lr_curve(experiments, output="show")