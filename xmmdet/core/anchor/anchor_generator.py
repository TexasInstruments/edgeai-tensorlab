import warnings
import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from mmdet.core.anchor.builder import ANCHOR_GENERATORS
from mmdet.core.anchor import AnchorGenerator

@ANCHOR_GENERATORS.register_module(force=True)
class SSDAnchorGenerator(AnchorGenerator):
    """Anchor generator for SSD

    Args:
        strides (list[int]  | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        basesize_ratio_range (tuple(float)): Ratio range of anchors.
        input_size (int): Size of feature map, 300 for SSD300,
            512 for SSD512.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. It is always set to be False in SSD.
    """

    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 input_size=300,
                 scale_major=True):
        assert len(strides) == len(ratios)
        assert mmcv.is_tuple_of(basesize_ratio_range, float)

        self.strides = [_pair(stride) for stride in strides]
        self.input_size = max(input_size) if isinstance(input_size, (list,tuple)) else input_size
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.basesize_ratio_range = basesize_ratio_range

        # calculate anchor ratios and sizes
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (self.num_levels - 2))
        min_sizes = []
        max_sizes = []
        for ratio in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(self.input_size * ratio / 100))
            max_sizes.append(int(self.input_size * (ratio + step) / 100))
        if self.input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(self.input_size * 7 / 100))
                max_sizes.insert(0, int(self.input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(self.input_size * 10 / 100))
                max_sizes.insert(0, int(self.input_size * 20 / 100))
            else:
                min_sizes.insert(0, int(self.input_size * basesize_ratio_range[0] * 0.4))
                max_sizes.insert(0, int(self.input_size * basesize_ratio_range[0]))
                warnings.warn(
                    'according to original SSD, basesize_ratio_range[0] should be either 0.15'
                    'or 0.2 when input_size is 300, got '
                    f'{basesize_ratio_range[0]}.')
        elif self.input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(self.input_size * 4 / 100))
                max_sizes.insert(0, int(self.input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(self.input_size * 7 / 100))
                max_sizes.insert(0, int(self.input_size * 15 / 100))
            else:
                min_sizes.insert(0, int(self.input_size * basesize_ratio_range[0] * 0.4))
                max_sizes.insert(0, int(self.input_size * basesize_ratio_range[0]))
                warnings.warn('according to original SSD, basesize_ratio_range[0] should be either 0.1'
                                 'or 0.15 when input_size is 512, got'
                                 f' {basesize_ratio_range[0]}.')
        else:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(self.input_size * 4 / 100))
                max_sizes.insert(0, int(self.input_size * 10 / 100))
            else:
                min_sizes.insert(0, int(self.input_size * basesize_ratio_range[0] * 0.4))
                max_sizes.insert(0, int(self.input_size * basesize_ratio_range[0]))

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 4 or 6 ratio
            anchor_ratios.append(torch.Tensor(anchor_ratio))
            anchor_scales.append(torch.Tensor(scales))

        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()
        # added for proto export
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios[i],
                center=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = torch.index_select(base_anchors, 0,
                                              torch.LongTensor(indices))
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}input_size={self.input_size},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}basesize_ratio_range='
        repr_str += f'{self.basesize_ratio_range})'
        return repr_str

