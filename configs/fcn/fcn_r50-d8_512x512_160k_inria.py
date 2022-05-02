_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/inria.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
