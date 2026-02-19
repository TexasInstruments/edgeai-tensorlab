from torchvision import models as tvmodels
from torch import nn
import re

def nested_getattr(obj, attr_path, default=None):
    from functools import reduce
    '''
        like getattr, but worked for nested objects. if attr_path = 'layer1.0.weight', it will result obj.layer1.0.weight
    '''
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default

def underscore_capitalize(string):
    # Input string with underscores

    # Capitalize each word and join with underscores
    capitalized_string = '_'.join(word.capitalize() for word in string.split('_'))
    return capitalized_string

def custom_capitalize(string):
    if not isinstance(string, str):
        return string
    # Capitalize after _
    capitalized_string = '_'.join(word.capitalize() for word in string.split('_'))
    # capitalize 'net' -> 'Net'
    capitalized_string = capitalized_string.replace('net', 'Net')
    capitalized_string = re.sub(r'fpn', 'FPN', capitalized_string, flags=re.IGNORECASE)
    # capitalized_string = capitalized_string.replace('next', 'NeXt')
    return capitalized_string

# NOTE: option should not be empty list, used for default option.
# Supported types: 'classification', 'detection', 'segmentation', 'video'
TVMODEL_NAMES = ['resnet', 'alexnet', 'convnext', 'densenet','mnasnet', 'efficientnet','efficientnetv2', 
                 'googlenet', 'inceptionv3', 'maxvit','mobilenet',
                 'regnet', 'resnext', 'shufflenet','squeezenet', 'vgg', 'wideresnet', 'swin', 'vit',
                 'fasterrcnn', 'fcos', 'retinanet', 'ssd', 'maskrcnn', 'keypointrcnn', 'deeplabv3', 
                 'fcn', 'lraspp', 'swin3d', 'resnet3d', 'mvit', 's3d'
                ]
TVMODEL_CONFIGS = {
    'resnet': {
        'type': 'classification',
        'class_fn': lambda option: f'resnet{option}',
        'options': [18,34,50,101,152],
        'weights_fn': lambda option: f'ResNet{option}_Weights.DEFAULT'
    },
    'alexnet': {
        'type': 'classification',
        'class_fn': lambda option: f'alexnet',
        'options': [''],
        'weights_fn': lambda option: f'AlexNet_Weights.DEFAULT'
    }, 
    'convnext': {
        'type': 'classification',
        'class_fn': lambda option: f'convnext_{option}',
        'options': ['tiny', 'small', 'base', 'large'],
        'weights_fn': lambda option: f'ConvNeXt_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'densenet': {
        'type': 'classification',
        'class_fn': lambda option: f'densenet{option}',
        'options': [121, 161, 169, 201],
        'weights_fn': lambda option: f'DenseNet{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'mnasnet': {
        'type': 'classification',
        'class_fn': lambda option: f'mnasnet{option}',
        'options':  ['0_5', '0_75', '1_0', '1_3'],
        'weights_fn': lambda option: f'MNASNet{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'efficientnet': {
        'type': 'classification',
        'class_fn': lambda option: f'efficientnet_b{option}',
        'options':  list(range(0,8)),
        'weights_fn': lambda option: f'EfficientNet_B{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'efficientnetv2': {
        'type': 'classification',
        'class_fn': lambda option: f'efficientnet_v2_{option}',
        'options':  ['s','m','l'] ,
        'weights_fn': lambda option: f'EfficientNet_V2_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'googlenet': {
        'type': 'classification',
        'class_fn': lambda option: f'googlenet{option}',
        'options':  [''] ,
        'weights_fn': lambda option: f'GoogLeNet{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'inceptionv3': {
        'type': 'classification',
        'class_fn': lambda option: f'inception_v3{option}',
        'options':  [''] ,
        'weights_fn': lambda option: f'Inception_V3{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'maxvit': {
        'type': 'classification',
        'class_fn': lambda option: f'maxvit_t{option}',
        'options':  [''] ,
        'weights_fn': lambda option: f'MaxVit_T{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'mobilenet': {
        'type': 'classification',
        'class_fn': lambda option: f'mobilenet_{option}',
        'options':  ['v2','v3_small', 'v3_large'] ,
        'weights_fn': lambda option: f'MobileNet_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'regnet': {
        'type': 'classification',
        'class_fn': lambda option: f'regnet_{option}',
        'options':  ['x_400mf', 'x_800mf', 'x_1_6gf', 'x_3_2gf', 'x_8gf', 'x_16gf', 'x_32gf', 'y_400mf', 'y_800mf', 'y_1_6gf', 'y_3_2gf', 'y_8gf', 'y_16gf', 'y_32gf', 'y_128gf'] ,
        'weights_fn': lambda option: f'RegNet_{option.upper()}_Weights.DEFAULT'
    },
    'resnext': {
        'type': 'classification',
        'class_fn': lambda option: f'resnext{option}',
        'options': ['50_32x4d','101_32x8d', '101_64x4d'],
        'weights_fn': lambda option: f'ResNeXt{option.upper()}_Weights.DEFAULT'
    },
    'shufflenet': {
        'type': 'classification',
        'class_fn': lambda option: f'shufflenet_v2_x{option}',
        'options':  ['0_5','1_0','1_5','2_0'],
        'weights_fn': lambda option: f'ShuffleNet_V2_X{option}_Weights.DEFAULT'
    },
    'squeezenet': {
        'type': 'classification',
        'class_fn': lambda option: f'squeezenet{option}',
        'options':  ['1_0','1_1'],
        'weights_fn': lambda option: f'SqueezeNet{option}_Weights.DEFAULT'
    },
    'swin': {
        'type': 'classification',
        'class_fn': lambda option: f'swin_{option}',
        'options':  ['t','s','b','v2_t','v2_s','v2_b'],
        'weights_fn': lambda option: f'Swin_{option.upper()}_Weights.DEFAULT'
    },
    'vgg': {
        'type': 'classification',
        'class_fn': lambda option: f'vgg{option}',
        'options':  ['11','11_bn','13','13_bn','16','16_bn','19','19_bn'],
        'weights_fn': lambda option: f'VGG{option.upper()}_Weights.DEFAULT'
    },
    'vit': {
        'type': 'classification',
        'class_fn': lambda option: f'vit_{option}',
        'options':  ['b_16', 'b_32', 'l_16', 'l_32', 'h_14'],
        'weights_fn': lambda option: f'ViT_{option.upper()}_Weights.DEFAULT'
    },
    'wideresnet': {
        'type': 'classification',
        'class_fn': lambda option: f'wide_resnet{option}',
        'options':  ['50_2', '101_2'] ,
        'weights_fn': lambda option: f'Wide_ResNet{option}_Weights.DEFAULT'
    },
    'fasterrcnn': {
        'type': 'detection',
        'class_fn': lambda option: f'fasterrcnn_{option}',
        'options':  ['resnet50_fpn', 'resnet50_fpn_v2', 'mobilenet_v3_large_fpn', 'mobilenet_v3_large_320_fpn'] ,
        'weights_fn': lambda option: f'FasterRCNN_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'fcos': {
        'type': 'detection',
        'class_fn': lambda option: f'fcos_{option}',
        'options':  ['resnet50_fpn'] ,
        'weights_fn': lambda option: f'FCOS_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'retinanet': {
        'type': 'detection',
        'class_fn': lambda option: f'retinanet_{option}',
        'options':  ['resnet50_fpn', 'resnet50_fpn_v2']    ,
        'weights_fn': lambda option: f'RetinaNet_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'ssd': {
        'type': 'detection',
        'class_fn': lambda option: f'ssd{option}',
        'options':  ['lite320_mobilenet_v3_large', '300_vgg16'],
        'weights_fn': lambda option: f'SSD{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'maskrcnn': {
        'type': 'detection',
        'class_fn': lambda option: f'maskrcnn_{option}',
        'options':  ['resnet50_fpn', 'resnet50_fpn_v2'] ,
        'weights_fn': lambda option: f'MaskRCNN_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'keypointrcnn': {
        'type': 'detection',
        'class_fn': lambda option: f'keypointrcnn_{option}',
        'options':  ['resnet50_fpn']  ,
        'weights_fn': lambda option: f'KeypointRCNN_{custom_capitalize(option)}_Weights.DEFAULT'
    },

    'deeplabv3': {
        'type': 'segmentation',
        'class_fn': lambda option: f'deeplabv3_{option}',
        'options':  ['mobilenet_v3_large', 'resnet50', 'resnet101'] ,
        'weights_fn': lambda option: f'DeepLabV3_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'fcn': {
        'type': 'segmentation',
        'class_fn': lambda option: f'fcn_{option}',
        'options':  ['resnet50', 'resnet101'],
        'weights_fn': lambda option: f'FCN_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'lraspp': {
        'type': 'segmentation',
        'class_fn': lambda option: f'lraspp_{option}',
        'options': ['mobilenet_v3_large']   ,
        'weights_fn': lambda option: f'LRASPP_{custom_capitalize(option)}_Weights.DEFAULT'
    },

    'swin3d': {
        'type': 'video',
        'class_fn': lambda option: f'swin3d_{option}',
        'options': ['t', 's', 'b'] ,
        'weights_fn': lambda option: f'Swin3D_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    'resnet3d': {
        'type': 'video',
        'class_fn': lambda option: f'{option}',
        'options': ['r3d_18', 'mc3_18', 'r2plus1d_18'],
        'weights_fn': lambda option: f'R3D_18_Weights.DEFAULT' if option == 'r3d_18' else None # TODO: implement
    },
    'mvit': {
        'type': 'video',
        'class_fn': lambda option: f'mvit_{option}',
        'options': ['v1_b', 'v2_s'],
        'weights_fn': lambda option: f'MViT_{custom_capitalize(option)}_Weights.DEFAULT'
    },
    's3d': {
        'type': 'video',
        'class_fn': lambda option: f's3d{option}',
        'options': [''],
        'weights_fn': lambda option: f'S3D{option}_Weights.DEFAULT'
    },


    
}


def construct_tv_model(model_name='resnet', verbose=False, **kwargs):
    """
    Docstring for construct_tv_model
    
    :param model_name: str
    :param kwargs: Accepted args:
        option: option for which subtype of model to use - typically depth/size etc. Default is first option in config
        fetch_weights: use pretrained weights (DEFAULT for now)
    """
    config = TVMODEL_CONFIGS[model_name]

    model_type = config.get('type', 'classification')
    option = kwargs.get('option', config['options'][0])
    
    if model_name.startswith('custom'):
        return config['class_fn'](option)
    
    prefix = '' if model_type == 'classification' else model_type+'.'
    model_path = prefix + config['class_fn'](option)
    
    if kwargs.get('fetch_weights', False):
        weights_path = prefix + config['weights_fn'](option)
        weights = nested_getattr(tvmodels, weights_path)
        if weights is None:
            raise ValueError(f'tvmodels.{weights_path} not found')
        else:
            if verbose:
                print(f'Using weights tvmodels.{weights_path}')
        return nested_getattr(tvmodels, model_path)(weights)
    
    return nested_getattr(tvmodels, model_path)()


def custom_swin(option):
    from torchvision.models.swin_transformer import _swin_transformer, SwinTransformerBlockV2, PatchMergingV2

    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2],
        num_heads=[1],
        window_size=[8, 8],
        stochastic_depth_prob=0,
        weights=None,
        progress=None,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )

    return model

def custom_resnet(option):
    model = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.DEFAULT)
    if option == 'base':
        return model
    elif option == 'kernel_11':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=3, bias=False)
    elif option == 'kernel_6':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=3, bias=False)
    elif option == 'kernel_12':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=3, bias=False)
    elif option == 'maxpool_5':
        model.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
    elif option == 'avgpool_5':
        model.maxpool = nn.AvgPool2d(kernel_size=5, stride=2, padding=1)
    elif option == 'gelu':
        model.relu = nn.GELU()
    elif option == 'hardactivation':
        model.relu = nn.SiLU()
        model.layer1.relu = nn.ReLU6()
        model.layer2.relu = nn.Hardswish()
        model.layer3.relu = nn.Hardsigmoid()
        model.layer4.relu = nn.LeakyReLU()
    return model

# custom models to test edge cases 
TVMODEL_CONFIGS.update({
    'custom_swin': {
        'type': 'classification',
        'class_fn': custom_swin,
        'options': ['']
    },   
    'custom_resnet': {
        'type': 'classification',
        'class_fn': custom_resnet,
        'options': ['base', 'kernel_11', 'kernel_12', 'kernel_6', 'maxpool_5', 'avgpool_5', 'gelu', 'hardactivation']
    },  
})
