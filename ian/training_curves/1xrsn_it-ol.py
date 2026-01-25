from pathlib import Path
from pose_estimation.analysis import Experiment
from matplotlib import pyplot as plt

experiments = [Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/",
                                        "ian_td-hm_rsn-ian_3xb128-360e_coco-256x192_it-ol"),
                          exp_times=["20260121_214515",
                                     "20260122_200638",
                                     "20260123_124145",
                                     "20260123_174904",
                                     "20260124_000245"],
                          legend="E:360 LR:4e-3 PolyLR"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_td-hm_rsn-ian_3xb128-180e_coco-256x192_ol"),
                          exp_times=["20260124_121658",
                                     "20260125_035745"],
                          legend="E:180 LR:4e-3 PolyLR"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_td-hm_rsn-ian_8xb32-210e_coco-256x192"),
                          exp_times=["20251126_173025"],
                          legend="E:210 LR:7.5e-3 MultiStepLR"),]

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_avg_loss_data, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_coco_ap_data, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.iter_lr_data, label=experiment.legend)
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.show()
