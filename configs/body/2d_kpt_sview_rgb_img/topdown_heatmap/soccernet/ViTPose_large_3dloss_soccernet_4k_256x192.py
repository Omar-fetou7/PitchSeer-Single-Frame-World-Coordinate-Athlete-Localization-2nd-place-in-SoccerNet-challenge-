_base_ = ['ViTPose_large_soccernet_4k_256x192.py']

# Load epoch_2 weights; reset epoch counter to 0.
# EDIT THIS — directory holding the trained Large heatmap-only checkpoint to fine-tune from
checkpoint_dir = '/path/to/checkpoints'
load_from = checkpoint_dir + '/vitpose_large_heatmap_epoch2.pth'

# 10x lower than original 4e-4 to avoid disrupting learned features.
optimizer = dict(lr=4e-5)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    min_lr_ratio=0.01)   # 4e-5 → 4e-7 smoothly

total_epochs = 10
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

# --- 3D projection loss ---
model = dict(
    loss_3d_keypoint=dict(
        type='LocSim3DLoss',
        keypoint_index=1,
        image_width=3840,
        image_height=2160,
        delta=0.5,
        temperature=100.0,
    ),
    lambda_3d=0.5,
)

# --- Pipeline with camera and orig bbox meta_keys ---
target_type = 'GaussianHeatmap'

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
            'rotation', 'bbox_score', 'flip_pairs',
            'camera_matrix', 'undist_poly', 'position_on_pitch',
            'orig_center', 'orig_scale',
        ]),
]

# EDIT THIS — directory containing images/{train,val,test}/ and annotations/{train,val,test}.json (COCO format)
data_root = '/path/to/data'
_ann_root = data_root + '/annotations'
_img_root = data_root + '/images'

data = dict(
    train=dict(
        type='TopDownSoccerNet3DDataset',
        ann_file=f'{_ann_root}/train.json',
        img_prefix=f'{_img_root}/train/',
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type='TopDownSoccerNet3DDataset',
        ann_file=f'{_ann_root}/val.json',
        img_prefix=f'{_img_root}/val/',
        dataset_info={{_base_.dataset_info}},
    ),
    test=dict(
        type='TopDownSoccerNet3DDataset',
        ann_file=f'{_ann_root}/test.json',
        img_prefix=f'{_img_root}/test/',
        dataset_info={{_base_.dataset_info}},
    ),
)
