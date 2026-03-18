"""Friday, 2026/03/13
Train the reimplemented HRNet, with an RSBAlterC-like block renamed to HRNetBlock"""

_base_ = ['../../../_base_/default_runtime.py']

custom_imports = dict(
    imports=['pose_estimation.models.backbones',
             'pose_estimation.models.blocks',
             'pose_estimation.models.structures',],
    allow_failed_imports=False
)

# runtime
train_cfg = dict(max_epochs=210, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam',
                                    lr=7.5e-3))

# learning policy
param_scheduler = [dict(type='LinearLR',
                        begin=0,
                        end=500,
                        start_factor=0.001,
                        by_epoch=False),
                   dict(type='MultiStepLR',
                        begin=0,
                        end=200,
                        milestones=[160, 190],
                        gamma=0.1,
                        by_epoch=True)]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=384)

# hooks
default_hooks = dict(checkpoint=dict(interval=5,
                                     max_keep_ckpts=3,
                                     save_last=True,
                                     save_best='coco/AP',
                                     rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNetRenewed',
        cfg_overrides={"level.block_name": "HRNetBlock",
                       "block.attention_name": "HRNetBlockAttention",
                       "block.base_branch_channels": 21,
                       "block_attention.enable_skip_connection": True,
                       "block_attention.attention_order": "parallel",
                       "block_attention.enable_channel_attention": True,
                       "block_attention.mlp_bottleneck": 8,
                       "block_attention.has_spatial_attention_phases": (False, True, True),
                       "block_attention.has_spatial_attention_norms": (False, True, True),
                       "block_attention.spatial_attention_map_channel_from": "all",
                       "block_attention.spatial_attention_map_channel_for": "group",},
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
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
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=48,
    num_workers=2,
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
    batch_size=48,
    num_workers=2,
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
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
