"""Evening, Saturday, April 18, 2026;
4xRSN-50-Ian as backbone;
RSB Alter E as block, ((3,), (3, 3,), (3, 3, 3,),);
Skip Connection around attention enabled (adaptive);
Channel Attention enabled, (Avg Pool, 1/8, ReLU, 8, Sigmoid);
Spatial Attention enabled, (DW 9x9 C2C, BN, ReLU, PW, ReLU, PW C2G, IN, Sigmoid);
Channel and Spatial Attention in parallel;
MMPose style Optimizer and LR Schedulers (Adam, 210 epochs, Step LR);
"""

_base_ = ['../../../_base_/default_runtime.py']

custom_imports = dict(
    imports=['pose_estimation.models.backbones',
             'pose_estimation.models.blocks',
             'pose_estimation.models.structures',],
    allow_failed_imports=False
)

# runtime
train_cfg = dict(max_epochs=360, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam',
                                    lr=4e-3,
                                    weight_decay=1e-5))

# learning policy
param_scheduler = [dict(type='LinearLR',
                        begin=0,
                        end=2400,
                        start_factor=0.1,
                        by_epoch=False),
                   dict(type='PolyLR',
                        eta_min=0.0,
                        power=1,
                        begin=0,
                        end=281160,
                        by_epoch=False)]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=384)

# hooks
default_hooks = dict(checkpoint=dict(interval=1,
                                     max_keep_ckpts=15,
                                     save_last=True,
                                     save_best='coco/AP',
                                     rule='greater'))

# codec settings
# multiple kernel_sizes of heatmap gaussian for 'Megvii' approach.
kernel_sizes = [15, 11, 9, 7, 5]
codec = [
    dict(
        type='MegviiHeatmap',
        input_size=(288, 384),
        heatmap_size=(72, 96),
        kernel_size=kernel_size) for kernel_size in kernel_sizes
]

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResidualStepsNetwork',
        stage_layer_blocks=((3, 4, 6, 3),
                            (3, 4, 6, 3),
                            (3, 4, 6, 3),
                            (3, 4, 6, 3),),
        cfg_overrides=dict({"down_layer.block_name": "RSBAlterC",
                            "block.relative_rfs": (2, 4, 6),
                            "block.attention_name": "AttentionForAblation",
                            "attention.enable_skip_connection": True,
                            "attention.use_centered_gating":  False,
                            "attention.attention_order": "parallel",
                            "attention.enable_channel_attention": True,
                            "attention.mlp_bottleneck": 8,
                            "attention.has_spatial_attention_phases": (False, True, True),
                            "attention.has_spatial_attention_norms": (False, True, True),
                            "attention.spatial_attention_map_channel_from": "all",
                            "attention.spatial_attention_map_channel_for": "group",}),
    ),
    head=dict(
        type='MSPNHead',
        out_shape=(96, 72),
        unit_channels=256,
        out_channels=17,
        num_stages=4,
        num_units=4,
        norm_cfg=dict(type='BN'),
        # each sub list is for a stage
        # and each element in each list is for a unit
        level_indices=[0, 1, 2, 3] * 3 + [1, 2, 3, 4],
        loss=([
            dict(
                type='KeypointMSELoss',
                use_target_weight=True,
                loss_weight=0.25)
        ] * 3 + [
            dict(
                type='KeypointOHKMMSELoss',
                use_target_weight=True,
                loss_weight=1.)
        ]) * 4,
        decoder=codec[-1]),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/data/ian/datasets/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec[0]['input_size']),
    dict(type='GenerateTarget', multilevel=True, encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec[0]['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none')
test_evaluator = val_evaluator

# fp16 settings
fp16 = dict(loss_scale='dynamic')
