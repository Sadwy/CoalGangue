# -----------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# -----------------------------------------------------------
# Modified by Sadwy
# -----------------------------------------------------------
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CoalGangueDataset(BaseSegDataset):
    """CoalGangue dataset.

    In segmentation map annotation for CoalGangue, 0 stands for background,
    which is included in 3 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'coal',
                 'gangue'),
        palette=[[120, 120, 120], [255, 255, 0],
                 [0, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
