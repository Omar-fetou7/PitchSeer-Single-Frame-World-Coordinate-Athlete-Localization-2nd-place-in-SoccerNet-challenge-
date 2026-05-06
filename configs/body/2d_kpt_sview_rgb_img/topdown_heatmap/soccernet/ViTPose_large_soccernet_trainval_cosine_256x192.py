_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/soccernet_4k.py',
]

evaluation = dict(interval=1, metric='mAP', save_best='AP')
checkpoint_config = dict(interval=1)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.8,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-2)

total_epochs = 5
target_type = 'GaussianHeatmap'

channel_cfg = dict(
    num_output_channels=2,
    dataset_joints=2,
    dataset_channel=[[0, 1]],
    inference_channel=[0, 1])

model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=1024,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    bbox_center_jitter_prob=0.0,
    bbox_center_jitter_factor=0.0,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownAffineMosaicAug',
        use_udp=True,
        bbox_jitter_prob=0.5,
        bbox_center_jitter=0.03,
        bbox_scale_jitter=0.05,
        rotation=5,
        rotation_prob=0.5,
        translate=0.1,
        translate_prob=0.5,
        scale=0.5,
        scale_prob=0.5,
        shear=1,
        shear_prob=0.5,
        hsv_prob=0.5,
        hue=0.015,
        saturation=0.7,
        value=0.4,
        mosaic_prob=0.0),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

# EDIT THIS — directory containing images/{train,val,test}/ and annotations/{train,val,test}.json (COCO format)
data_root = '/path/to/data'
# EDIT THIS — directory containing the upstream ViTPose++ pretrained weights (split via tools/model_split.py)
pretrained_dir = '/path/to/pretrained'

ann_root = data_root + '/annotations'
images_root = data_root + '/images'

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=[
        dict(
            type='TopDownCocoDataset',
            ann_file=f'{ann_root}/train.json',
            img_prefix=f'{images_root}/train/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownCocoDataset',
            ann_file=f'{ann_root}/val.json',
            img_prefix=f'{images_root}/val/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
    ],
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{ann_root}/test.json',
        img_prefix=f'{images_root}/test/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{ann_root}/test.json',
        img_prefix=f'{images_root}/test/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

load_from = pretrained_dir + '/coco.pth'
