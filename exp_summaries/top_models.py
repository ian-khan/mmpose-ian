from pathlib import Path
from pose_estimation.analysis import Experiment
from matplotlib import pyplot as plt

experiments = [Experiment(work_dir=Path("/work_dirs/ian_td-hm_4xrsn-alter-c_3xb64-210e_coco-384x288"),
                          exp_times=["20260112_154914"],
                          label="4xRSN-50 Alter C"),
               Experiment(work_dir=Path("/work_dirs/ian_td-hm_hrformer-base_8xb32-210e_coco-384x288"),
                          exp_times=["20251003_173749"],
                          label="HRFormer-B"),
               Experiment(work_dir=Path("/work_dirs/ian_td-hm_swin-l-p4-w12_8xb32-210e_coco-384x288"),
                          exp_times=["20251103_130755",
                                     "20251104_195534",
                                     "20251105_120356",
                                     "20251107_174406",
                                     "20251109_143439"],
                          label="Swin Transformer-L"),
               ]

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_avg_loss_data, label=experiment.label)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("260201_loss.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.epoch_coco_ap_data, label=experiment.label)
plt.xlabel("Epoch")
plt.ylabel("COCO AP")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("260201_ap.jpg")

plt.figure()
for experiment in experiments:
    plt.plot(*experiment.iter_lr_data, label=experiment.label)
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("260201_lr.jpg")