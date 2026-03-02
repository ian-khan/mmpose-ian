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

experiments = [Experiment(file_name="/work_dirs/ian_td-hm_rsn-ian_8xb32-210e_coco-256x192/20251126_173025/vis_data/20251126_173025.json",
                          legend="Ian Baseline"),
               Experiment(file_name="/work_dirs/ian_td-hm_rsn-ian_3xb512-200e_coco-256x192_optim-schdl/20260117_043533/vis_data/20260117_043533.json",
                          legend="Optimizer Scheduler"),]

plt.figure()
for experiment in experiments:
    plt.plot(experiment.loss_epoch_x, experiment.loss_epoch_y, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("1xrsn-repro-loss.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(experiment.ap_epoch_x, experiment.ap_epoch_y, label=experiment.legend)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
# plt.ylim((0.65, 0.75))
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("1xrsn-repro-ap.jpg")