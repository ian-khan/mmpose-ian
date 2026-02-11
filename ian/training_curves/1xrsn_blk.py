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
                          label="RSB-Ian"),
               Experiment(work_dir=Path("/data/ian/remote_projects/mmpose/work_dirs/"
                                        "ian_ab_td-hm_rsn-alter-c_3xb128-360e_coco-256x192"),
                          exp_times=["20260127_133523"],
                          label="RSB Alter C"),
               ]

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_avg_loss_data, label=experiment.label)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("260130_loss.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_coco_ap_data, label=experiment.label)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("260130_ap.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.iter_lr_data, label=experiment.label)
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("260130_lr.jpg")
