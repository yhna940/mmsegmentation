_base_ = [
    'upernet_swin_tiny_patch4_window7_512x512_160k_inria_'
    'pretrain_224x224_1K.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=2),
    auxiliary_head=dict(in_channels=512, num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
