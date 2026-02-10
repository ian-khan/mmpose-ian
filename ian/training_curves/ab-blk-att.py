from pathlib import Path
from pose_estimation.analysis import Experiment
from matplotlib import pyplot as plt

experiments = [Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-0_td-hm_rsn50-d_3xb128-200e_coco-256x192"),
                          exp_times=["20260206_193130"],
                          legend="test 0"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab-blk-att-1_td-hm_rsn50-c_3xb128-200e_coco-256x192"),
                          exp_times=["20260208_043650"],
                          legend="test 1"),
               ]

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_avg_loss_data, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("260127_loss.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_coco_ap_data, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("260127_ap.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.iter_lr_data, label=experiment.legend)
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("260127_step_lr.jpg")