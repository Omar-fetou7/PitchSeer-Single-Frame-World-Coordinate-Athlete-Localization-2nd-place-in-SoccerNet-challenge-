_base_ = ['ViTPose_small_simple_soccernet_4k_256x192.py']

# --- Training schedule ---
# Load epoch_30 weights (already converged on heatmap loss); reset epoch counter.
# Use 10× lower lr than original 5e-4 to avoid disrupting learned features.

# EDIT THIS — directory holding the trained ViTPose checkpoint (e.g. downloaded from HuggingFace)
checkpoint_dir = '/path/to/checkpoints'
load_from = checkpoint_dir + '/vitpose_small_baseline_epoch30.pth'

optimizer = dict(lr=5e-5)

lr_config = dict(
    _delete_=True,         # drop parent's step=[35,45] — incompatible with CosineAnnealing
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    min_lr_ratio=0.01)   # 5e-5 → 5e-7 smoothly over 15 epochs

total_epochs = 15
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

# --- 3D projection loss ---
# Added to model as optional params (backward-compatible with old configs).
# lambda_3d=0.001 makes the 3D loss contribution comparable in magnitude to
# the heatmap loss plateau (~0.0003).
model = dict(
    loss_3d_keypoint=dict(
        type='LocSim3DLoss',
        keypoint_index=1,
        image_width=3840,
        image_height=2160,
        delta=0.5,
        temperature=100.0,
    ),
    lambda_3d=1,
)

# --- Pipeline with camera meta_keys ---
# Redefine train_pipeline to include camera/position fields via Collect.
# val/test pipelines don't need them (loss is only during training).
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

# --- Dataset class with camera fields and absolute data paths ---
# EDIT THIS — directory containing images/{train,val,test}/ and annotations/{train,val,test}.json (COCO format)
data_root = '/path/to/data'
_ann_root = data_root + '/annotations'
_img_root = data_root + '/images'

data = dict(
    samples_per_gpu=128,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
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
        ann_file=f'{_ann_root}/val.json',
        img_prefix=f'{_img_root}/val/',
        dataset_info={{_base_.dataset_info}},
    ),
)
