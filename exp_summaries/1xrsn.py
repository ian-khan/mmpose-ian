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
                                    "ian_td-hm_rsn-ian_8xb32-210e_coco-256x192/"
                                    "20251126_173025/vis_data/20251126_173025.json",
                          legend="Ian 5e-3"),
               Experiment(file_name="/data/ian/remote_projects/mmpose/work_dirs/"
                                    "ian_td-hm_rsn-alter-a_4xb64-210e_coco-256x192/"
                                    "20251226_214118/vis_data/20251226_214118.json",
                          legend="Alter-A 5e-3"),
               Experiment(file_name="/data/ian/remote_projects/mmpose/work_dirs/"
                                    "ian_td-hm_rsn-alter-b_8xb32-210e_coco-256x192/"
                                    "20260103_194352/vis_data/20260103_194352.json",
                          legend="Alter-B; 0,2,4; 3; 5e-3"),
               Experiment(file_name="/data/ian/remote_projects/mmpose/work_dirs/"
                                    "ian_td-hm_rsn-alter-b_8xb32-210e_coco-256x192/"
                                    "20260107_052034/vis_data/20260107_052034.json",
                          legend="Alter-B; 0,2,6; 4; 5e-3"),
               Experiment(file_name="/data/ian/remote_projects/mmpose/work_dirs/"
                                    "ian_td-hm_rsn-alter-b_8xb32-210e_coco-256x192/"
                                    "20260108_154429/vis_data/20260108_154429.json",
                          legend="Alter-B; 0,2,6; 4; 1e-3"),]

plt.figure()
for experiment in experiments:
    plt.plot(experiment.loss_epoch_x, experiment.loss_epoch_y, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("1xrsn-loss.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(experiment.ap_epoch_x, experiment.ap_epoch_y, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
plt.ylim((0.65, 0.75))
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("1xrsn-ap-close.jpg")