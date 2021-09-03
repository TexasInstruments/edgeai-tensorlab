from collections import OrderedDict
import torch
from .._utils import IntermediateLayerGetter
from ..._internally_replaced_utils import load_state_dict_from_url
from .. import mobilenetv3
from .. import mobilenet
from .. import resnet
from .deeplabv3 import DeepLabHead, DeepLabV3
from .deeplabv3plus import DeepLabV3PlusHead, DeepLabV3Plus
from .fcn import FCN, FCNHead
from .lraspp import LRASPP
from ... import xnn


__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
           'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large',
           # lite models
           'lraspp_mobilenet_v3_lite_large', 'deeplabv3_mobilenet_v3_lite_large',
           'deeplabv3plus_mobilenet_v3_lite_large', 'deeplabv3plus_mobilenet_v3_lite_small',
           'deeplabv3plus_mobilenet_v2_lite']


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco':
        'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
    'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth',
}


def _segm_model(name, backbone_name, num_classes, aux=False, pretrained_backbone=True, conv_cfg=None, skip_tail=False):
    shortcut_name = None
    shortcut_channels = None
    if 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
        if name == 'deeplabv3plus':
            shortcut_name = 'layer1'
            shortcut_channels = 256
        #
    elif 'mobilenet_v3' in backbone_name or 'mobilenet_v2' in backbone_name:
        backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features
        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = (len(backbone)-2) if skip_tail else stage_indices[-1] # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
        if name == 'deeplabv3plus':
            shortcut_pos = 3
            shortcut_name = str(shortcut_pos)
            shortcut_channels = backbone[shortcut_pos].out_channels
        #
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    aux_classifier = None
    return_layers = OrderedDict()
    if shortcut_channels:
        return_layers[shortcut_name] = 'shortcut'
    #
    return_layers[out_layer] = 'out'
    if aux:
        return_layers[aux_layer] = 'aux'
        aux_classifier = FCNHead(aux_inplanes, num_classes)
    #
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'deeplabv3plus': (DeepLabV3PlusHead, DeepLabV3Plus),	
        'fcn': (FCNHead, FCN),
    }
    classifier = model_map[name][0](out_inplanes, num_classes=num_classes,
                                    shortcut_channels=shortcut_channels, conv_cfg=conv_cfg)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss=True, **kwargs):
    if pretrained:
        kwargs["pretrained_backbone"] = False
    model = _segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress, pretrained)
    return model


def _load_state_dict(model, state_dict):
    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict(state_dict, strict=False)


def _load_weights(model, arch_type, backbone, progress, pretrained=None):
    arch = arch_type + '_' + backbone + '_coco'
    if pretrained is True:
        if model_urls.get(arch, None) is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        _load_state_dict(model, state_dict)
    elif xnn.utils.is_url(pretrained):
        state_dict = load_state_dict_from_url(pretrained, progress=progress)
        _load_state_dict(model, state_dict)
    elif isinstance(pretrained, str):
        state_dict = torch.load(pretrained)
        _load_state_dict(model, state_dict)


def _segm_lraspp_mobilenetv3(backbone_name, num_classes, pretrained_backbone=True):
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): 'low', str(high_pos): 'high'})

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model


def fcn_resnet50(pretrained=False, progress=True,
                 num_classes=21, aux_loss=None, pretrained_backbone=True, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('fcn', 'resnet50', pretrained, progress, num_classes, aux_loss,
                       pretrained_backbone=pretrained_backbone, **kwargs)


def fcn_resnet101(pretrained=False, progress=True,
                  num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('fcn', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_mobilenet_v3_large(pretrained=False, progress=True,
                                 num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrained, progress, num_classes, aux_loss, **kwargs)


def lraspp_mobilenet_v3_large(pretrained=False, progress=True, num_classes=21, **kwargs):
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError('This model does not use auxiliary loss')

    backbone_name = 'mobilenet_v3_large'
    model = _segm_lraspp_mobilenetv3(backbone_name, num_classes, **kwargs)

    if pretrained:
        _load_weights(model, 'lraspp', backbone_name, progress, pretrained)

    return model


#################################################################################################
def lraspp_mobilenet_v3_lite_large(pretrained=False, progress=True,
                                   num_classes=21, backbone_name='mobilenet_v3_lite_large', **kwargs):
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError('This model does not use auxiliary loss')

    model = _segm_lraspp_mobilenetv3(backbone_name, num_classes, **kwargs)

    if pretrained:
        _load_weights(model, 'lraspp', backbone_name, progress, pretrained)

    return model


def deeplabv3_mobilenet_v3_lite_large(pretrained=False, progress=True,
                    num_classes=21, backbone_name='mobilenet_v3_lite_large', **kwargs):
    '''DeepLabV3 with MobileNetV3 Large Backbone - using Depthwise separable layers'''
    return _load_model('deeplabv3', backbone_name, pretrained, progress, num_classes, conv_cfg=dict(group_size_dw=1), skip_tail=True, **kwargs)


def deeplabv3plus_mobilenet_v2_lite(pretrained=False, progress=True,
                    num_classes=21, backbone_name='mobilenet_v2_lite', **kwargs):
    '''DeepLabV3Plus with MobileNetV2 Backbone - using Depthwise separable layers'''
    return _load_model('deeplabv3plus', backbone_name, pretrained, progress, num_classes, conv_cfg=dict(group_size_dw=1), skip_tail=True, **kwargs)


def deeplabv3plus_mobilenet_v3_lite_large(pretrained=False, progress=True,
                    num_classes=21, backbone_name='mobilenet_v3_lite_large', **kwargs):
    '''DeepLabV3Plus with MobileNetV3 Large Backbone - using Depthwise separable layers'''
    return _load_model('deeplabv3plus', backbone_name, pretrained, progress, num_classes, conv_cfg=dict(group_size_dw=1), skip_tail=True, **kwargs)


def deeplabv3plus_mobilenet_v3_lite_small(pretrained=False, progress=True,
                    num_classes=21, backbone_name='mobilenet_v3_lite_small', **kwargs):
    '''DeepLabV3Plus with MobileNetV3 Small Backbone - using Depthwise separable layers'''
    return _load_model('deeplabv3plus', backbone_name, pretrained, progress, num_classes, conv_cfg=dict(group_size_dw=1), skip_tail=True, **kwargs)
