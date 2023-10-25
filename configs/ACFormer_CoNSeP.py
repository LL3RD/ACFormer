mmdet_baseline = "../thirdparty/mmdetection/configs/cell_det"
_base_ = [
    f"{mmdet_baseline}/deformable_detr_rf_ts_Convnext_256x256.py",
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth"

model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    bbox_head=dict(
        transformer=dict(encoder=dict(num_layers=3, ), decoder=dict(num_layers=3, )),
        loss_cls=dict(
            type='FocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_point=dict(loss_weight=5.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner_CellDet',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='PointL1Cost', weight=5.0))
    )
)

global_pipeline = [
    dict(type="AddPointsAttr"),
    dict(type="ExtraAttrs", tag="global"),
    dict(type="CellFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_points"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

local_pipeline = [
    dict(type="AddPointsAttr"),
    dict(type="ExtraAttrs", tag="local"),
    dict(type="CellFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_points", "ori_img"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AddOriImg'),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[
                        (672, 672), (704, 704), (736, 736),
                        (768, 768), (800, 800), (832, 832), (864, 864), (900, 900)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(800, 800)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(800, 800),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[
                        (672, 672), (704, 704), (736, 736),
                        (768, 768), (800, 800), (832, 832), (864, 864), (900, 900)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type="Pad", size_divisor=1),
    dict(type="Normalize", **img_norm_cfg),

    dict(type="MultiBranch", globalb=local_pipeline),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

dataset_type = 'CellDetDataset_CoNSeP_Best'
data_root = "/data2/huangjunjia/coco/COCO_CoNSeP_256/"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco_CoNSeP_train.json',
        img_prefix=data_root + 'CoNSeP_Train/',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco_CoNSeP_test.json',
        img_prefix=data_root + 'CoNSeP_Test/',
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco_CoNSeP_test.json',
        img_prefix=data_root + 'CoNSeP_Test/',
        pipeline=test_pipeline,
        data_root=data_root))

semi_wrapper = dict(
    type="GlobalLocal_STN_Sequence",
    model="${model}",
    train_cfg=dict(
        pseudo_label_initial_score_thr=0.1,
        global_weight=0,
    ),
    test_cfg=dict(inference_on="locals"),
    stn_cfg=dict(
        in_chan=3,
        embed=120,
    )
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanGlobal", momentum=0.999, interval=1, warm_up=0),
    dict(type="GlobalWeightStep", value=0.5, step=48000),
    # dict(type='GlobalWeight', max_value=2)
]
evaluation = dict(type="SubModulesDistEvalHook", interval=800)
# load_from = "/mntnfs/med_data2/huangjunjia/mmdet/consep/goballocal_STNSequence_TrainFromScratch_V2/iter_32000.pth"
# load_from = "/mntnfs/med_data2/huangjunjia/mmdet/consep/goballocal_STNSequence_TrainFromScratch_Global/iter_16000.pth"
# find_unused_parameters = True

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            'locals.backbone': dict(lr_mult=0.1),
            'locals.sampling_offsets': dict(lr_mult=0.1),
            'locals.reference_points': dict(lr_mult=0.1),
            'transformer': dict(lr_mult=0.1),
            'fc_loc': dict(lr_mult=0.1),
            # 'patch_embed': dict(lr_mult=0.0),
            # 'absolute_pos_embed': dict(lr_mult=0.0),
            # 'affine_token': dict(lr_mult=0.0),
        }))
