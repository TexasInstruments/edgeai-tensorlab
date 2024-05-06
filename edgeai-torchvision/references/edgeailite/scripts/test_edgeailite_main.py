#!/usr/bin/env python

import torchvision

################################################################
print('torchvision models')
model_list = [m for m in dir(torchvision.models) if not m.startswith('__')]
print('torchvision backbone models', model_list)
model = torchvision.models.mobilenet_v2(pretrained=False)
# this is same as above, just another syntax of the same thing
model = torchvision.models.__dict__['mobilenet_v2'](pretrained=False)


model_list = [m for m in dir(torchvision.models.segmentation) if not m.startswith('__')]
print('torchvision segmentation models', model_list)
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, pretrained_backbone=False)
model = torchvision.models.segmentation.lraspp_mobilenet_v3_lite_large(pretrained=False, pretrained_backbone=False)


model_list = [m for m in dir(torchvision.models.detection) if not m.startswith('__')]
print('torchvision detection models', model_list)
model = torchvision.models.detection.ssd300_vgg16(pretrained=False, pretrained_backbone=False)
model = torchvision.models.detection.ssdlite_mobilenet_v3_lite_large(pretrained=False, pretrained_backbone=False)


################################################################
print('torchvision.edgeailite models')
model_list = [m for m in dir(torchvision.edgeailite.xvision.models) if not m.startswith('__')]
print('torchvision.edgeailite backbone models', model_list)
model = torchvision.edgeailite.xvision.models.mobilenetv2_x1(pretrained=False)
model = torchvision.edgeailite.xvision.models.mobilenetv3_lite_large_x1(pretrained=False)

model_list = [m for m in dir(torchvision.edgeailite.xvision.models.pixel2pixel) if not m.startswith('__')]
print('torchvision.edgeailite pixel2pixel models', model_list)
model_config = dict(output_type=['segmentation'], output_channels=[19])
model = torchvision.edgeailite.xvision.models.pixel2pixel.fpn_mobilenetv2_tv_edgeailite(model_config)