_base_ = [
    '../_base_/datasets/waymo-3d-3class-3sweep.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

voxel_size = (0.32, 0.32, 6)
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]

model = dict(
    type='DynamicCenterPoint',

    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='FlatFormer',
        in_channels=128,
        num_heads=8,
        num_blocks=2,
        activation="gelu",
        window_shape=(13, 13, 1),
        sparse_shape=(468, 468, 1),
        output_shape=(468, 468),
        pos_temperature=10000,
        normalize_pos=False,
        group_size=144,
    ),

    backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[64, 128],
        layer_nums=[3, 3],
        layer_strides=[1, 2],
        conv_cfg=dict(type='Conv2d', bias=False),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        residual=True,
    ),

    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        out_channels=[128, 128]
    ),

    bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2), iou=(1, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-74.88, -74.88, -10.0, 74.88, 74.88, 10.0],
            max_num=4096,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3,
        ),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True
    ),

    # model training and testing settings
    train_cfg=dict(
        grid_size=[468, 468, 1],
        voxel_size=voxel_size,
        out_size_factor=1,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=1,
        point_cloud_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        iou_weight=1.0
    ),

    test_cfg=dict(
        post_center_limit_range=[-80, -80, -10, 80, 80, 10],
        max_per_img=500, # what is this
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175], # not used in normal nms, task-wise
        score_threshold=0.1,
        pc_range=point_cloud_range[:2], # seems not used
        out_size_factor=1,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=[2048, 1024, 1024],
        post_max_size=[300, 100, 100],
        nms_thr=[0.8, 0.55, 0.55],
        iou_pow=2.0
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=1)
    ),
)
