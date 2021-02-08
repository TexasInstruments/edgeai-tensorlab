
# for tensorflow object detection api models that are trained with 1+90 classes.
# Note: typically other training frameworks use 1+80 classes - because only 80 classes are annotated in coco.
coco_91class_background_label_id = 0
coco_91class_background_label_offset = (0 if coco_91class_background_label_id == 0 else 1)
coco_91class_label_offset = {i:(coco_91class_background_label_offset+i) for i in range(91)}


# for github.com/google/automl models (tensorflow) that are trained with 90 classes.
# Note: typically other training frameworks use 1+80 classes - because only 80 classes are annotated in coco.
coco_90class_background_label_id = -1
coco_90class_background_label_offset = (0 if coco_90class_background_label_id == 0 else 1)
coco_90class_label_offset = {i:(coco_90class_background_label_offset+i) for i in range(90)}


tfLiteMlperfSSDMobileNetV1OutNames1200x1200 = \
    '"BoxPredictor_0/BoxEncodingPredictor/BiasAdd,BoxPredictor_0/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_1/BoxEncodingPredictor/BiasAdd,BoxPredictor_1/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_2/BoxEncodingPredictor/BiasAdd,BoxPredictor_2/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_3/BoxEncodingPredictor/BiasAdd,BoxPredictor_3/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_4/BoxEncodingPredictor/BiasAdd,BoxPredictor_4/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_5/BoxEncodingPredictor/BiasAdd, BoxPredictor_5/ClassPredictor/BiasAdd"'

tfLiteMlperfSSDMobileNetV2OutNames300x300 = \
    '"BoxPredictor_0/BoxEncodingPredictor/BiasAdd;BoxPredictor_0/BoxEncodingPredictor/Conv2D;BoxPredictor_0/BoxEncodingPredictor/biases1, ' \
    'BoxPredictor_0/ClassPredictor/BiasAdd;BoxPredictor_0/ClassPredictor/Conv2D;BoxPredictor_0/ClassPredictor/biases1, ' \
    'BoxPredictor_1/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_1/BoxEncodingPredictor/Conv2D;BoxPredictor_1/BoxEncodingPredictor/biases1, ' \
    'BoxPredictor_1/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_1/ClassPredictor/Conv2D;BoxPredictor_1/ClassPredictor/biases1, ' \
    'BoxPredictor_2/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_2/BoxEncodingPredictor/Conv2D;BoxPredictor_2/BoxEncodingPredictor/biases1, ' \
    'BoxPredictor_2/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_2/ClassPredictor/Conv2D;BoxPredictor_2/ClassPredictor/biases1, ' \
    'BoxPredictor_3/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_3/BoxEncodingPredictor/Conv2D;BoxPredictor_3/BoxEncodingPredictor/biases1, ' \
    'BoxPredictor_3/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_3/ClassPredictor/Conv2D;BoxPredictor_3/ClassPredictor/biases1, '\
    'BoxPredictor_4/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_4/BoxEncodingPredictor/Conv2D;BoxPredictor_4/BoxEncodingPredictor/biases1, ' \
    'BoxPredictor_4/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_4/ClassPredictor/Conv2D;BoxPredictor_4/ClassPredictor/biases1, ' \
    'BoxPredictor_5/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_5/BoxEncodingPredictor/biases1, ' \
    'BoxPredictor_5/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_5/ClassPredictor/biases1"'

tfLiteSSDLiteMobileNetV2OutNames300x300 = \
    '"BoxPredictor_0/BoxEncodingPredictor/BiasAdd, ' \
    'BoxPredictor_0/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_1/BoxEncodingPredictor/BiasAdd, ' \
    'BoxPredictor_1/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_2/BoxEncodingPredictor/BiasAdd, ' \
    'BoxPredictor_2/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_3/BoxEncodingPredictor/BiasAdd, ' \
    'BoxPredictor_3/ClassPredictor/BiasAdd, '\
    'BoxPredictor_4/BoxEncodingPredictor/BiasAdd, ' \
    'BoxPredictor_4/ClassPredictor/BiasAdd, ' \
    'BoxPredictor_5/BoxEncodingPredictor/BiasAdd, ' \
    'BoxPredictor_5/ClassPredictor/BiasAdd"'

tf2LiteSSDMobileNetV2OutDataNames320x320 = \
    '"BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/BiasAdd;BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/BiasAdd;BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalBoxHead_1/BoxEncodingPredictor/BiasAdd;BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_1/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_1/BoxEncodingPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalClassHead_1/ClassPredictor/BiasAdd;BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_1/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_1/ClassPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalBoxHead_2/BoxEncodingPredictor/BiasAdd;BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_2/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_2/BoxEncodingPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalClassHead_2/ClassPredictor/BiasAdd;BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_2/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_2/ClassPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalBoxHead_3/BoxEncodingPredictor/BiasAdd;BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_3/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_3/BoxEncodingPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalClassHead_3/ClassPredictor/BiasAdd;BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_3/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_3/ClassPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalBoxHead_4/BoxEncodingPredictor/BiasAdd;BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_4/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_4/BoxEncodingPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalClassHead_4/ClassPredictor/BiasAdd;BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_4/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_4/ClassPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/BiasAdd;BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/Conv2D;BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/bias1, ' \
    'BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/BiasAdd;BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/Conv2D;BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/bias1"'

tf2LiteSSDMobileNetV2FPNOutDataNames640x640 = \
    '"WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/BiasAdd;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/BiasAdd;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/BiasAdd_1;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_1;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/BiasAdd_1;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_1;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/BiasAdd_2;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_2;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/BiasAdd_2;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_2;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/BiasAdd_3;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_3;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/BiasAdd_3;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_3;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/bias, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/BiasAdd_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead/BoxPredictor/bias1, ' \
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/BiasAdd_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/separable_conv2d_4;WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead/ClassPredictor/bias1"'


githubOnnxYolov3OutDataNames416x416 = \
    '"convolution_output, ' \
    'convolution_output1, ' \
    'convolution_outpu2"'


tf2LiteEfficientDetLite0OutDataNames512x512 = \
    '"class_net/class-predict/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict/separable_conv2d;class_net/class-predict/bias, ' \
    'box_net/box-predict/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict/separable_conv2d;box_net/box-predict/bias, ' \
    'class_net/class-predict_1/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_1/separable_conv2d;class_net/class-predict/bias, ' \
    'box_net/box-predict_1/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_1/separable_conv2d;box_net/box-predict/bias, ' \
    'class_net/class-predict_2/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_2/separable_conv2d;class_net/class-predict/bias, ' \
    'box_net/box-predict_2/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_2/separable_conv2d;box_net/box-predict/bias, ' \
    'class_net/class-predict_3/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_3/separable_conv2d;class_net/class-predict/bias, ' \
    'box_net/box-predict_3/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_3/separable_conv2d;box_net/box-predict/bias, ' \
    'class_net/class-predict_4/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict/bias1, ' \
    'box_net/box-predict_4/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict/bias1"'

