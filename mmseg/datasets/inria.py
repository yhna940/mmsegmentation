# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class InriaDataset(CustomDataset):
    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(InriaDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
