
https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py

preprocess_image:

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_mean_image_subtraction(image, CHANNEL_MEANS, num_channels)

No Scaling is done.
