from pose_estimation.performance.training_curves import get_training_curves_data
from matplotlib import pyplot as plt
from pathlib import Path

class Experiment:
    def __init__(self,
                 file_name: str,
                 legend: str):
        self.file_name = Path(file_name)
        self.legend = legend
        self.loss_epoch_x, self.loss_epoch_y, self.ap_epoch_x, self.ap_epoch_y = get_training_curves_data(self.file_name)

experiments = [Experiment(file_name="/data/ian/remote_projects/mmpose/work_dirs/"
                                    "ian_td-hm_3xrsn50_8xb32-210e_coco-256x192/"
                                    "20250922_092011/vis_data/20250922_092011.json",
                          legend="3xRSN-50"),
               Experiment(file_name="/data/ian/remote_projects/mmpose/work_dirs/"
                                    "ian_td-hm_3xrsn-alter-c_8xb32-210e_coco-256x192/"
                                    "20260109_224458/vis_data/20260109_224458.json",
                          legend="3xRSN-50 Alter C"),]

plt.figure()
for experiment in experiments:
    plt.plot(experiment.loss_epoch_x, experiment.loss_epoch_y, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("3xrsn-loss.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(experiment.ap_epoch_x, experiment.ap_epoch_y, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("3xrsn-coco-ap.jpg")