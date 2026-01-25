# Config for RSN-Ian, using the same iterations; optimizer and LR scheduler as the RSN paper

_base_ = ['../../../_base_/default_runtime.py']

custom_imports = dict(
    imports=['pose_estimation.models.backbones.rsn_ian',
             'pose_estimation.models.blocks.cnn_blocks'],
    allow_failed_imports=False
)

# runtime
train_cfg = dict(max_epochs=180, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam',
                                    lr=5e-4,
                                    weight_decay=1e-5))

# learning policy
param_scheduler = [dict(type='LinearLR',
                        begin=0,
                        end=1200,
                        start_factor=0.1,
                        by_epoch=False),
                   dict(type='PolyLR',
                        eta_min=0.0,
                        power=1,
                        begin=0,
                        end=70380,
                        by_epoch=False)]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=48)

# hooks
default_hooks = dict(checkpoint=dict(interval=5,
                                     max_keep_ckpts=6,
                                     save_last=True,
                                     save_best='coco/AP',
                                     rule='greater'))

# codec settings
# multiple kernel_sizes of heatmap gaussian for 'Megvii' approach.
kernel_sizes = [11, 9, 7, 5]
codec = [
    dict(
        type='MegviiHeatmap',
        input_size=(192, 256),
        heatmap_size=(48, 64),
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
        stage_layer_blocks=((3, 4, 6, 3),),
        cfg_overrides={"down_layer.block_name": "RSBIan",
                       "block.base_branch_channels": 26,
                       "block.relative_rfs": ("",) * 4,},
    ),
    head=dict(
        type='MSPNHead',
        out_shape=(64, 48),
        unit_channels=256,
        out_channels=17,
        num_stages=1,
        num_units=4,
        norm_cfg=dict(type='BN'),
        # each sub list is for a stage
        # and each element in each list is for a unit
        level_indices=[0, 1, 2, 3],
        loss=[
            dict(
                type='KeypointMSELoss',
                use_target_weight=True,
                loss_weight=0.25)
        ] * 3 + [
            dict(
                type='KeypointOHKMMSELoss',
                use_target_weight=True,
                loss_weight=1.)
        ],
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
    batch_size=128,
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
    batch_size=128,
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
