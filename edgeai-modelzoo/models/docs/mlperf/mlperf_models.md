# MLPerf - Model Conversion scripts

Please refer to the classification and detection section in the MLPerf inference repository for detailed description of the models: [MLPerf/inference/vision/classification and detection](https://github.com/mlperf/inference/tree/r0.7/vision/classification_and_detection)


<hr>

## MobileNet Classification Model
#### Model links
- https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz
- https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx

#### How to use in TIDL
This model is provided in various formats including TFLite and ONNX - any of these formats can be used in TIDL.

#### Training Repository
- [tensorflow 1.0 slim](https://github.com/tensorflow/models/tree/r1.13.0/research/slim)


<hr>

## ResNet50 Classification Model
#### Model links
- https://zenodo.org/record/2535873/files/resnet50_v1.pb
- https://zenodo.org/record/2592612/files/resnet50_v1.onnx

#### Training Repository
- [tensorflow 1.0 resnet models](https://github.com/tensorflow/models/tree/r1.13.0/official/resnet)
- [tensorflow 1.0 resnet models - copy at mlperf training repository](https://github.com/mlperf/training/tree/master/image_classification)


#### Convert the model to use in TIDL
This model has a few layers at the end that TIDL doesn't support currently. However, converting the model to TFLite format will remove those unsupported layers. The following steps show example code to convert the model to TFLite format.

```
graph_def_file = '/data/tensorlabdata1/modelzoo/mlperf/files/image_classification/resnet50_v1_tf.pb'
output_file = '/data/tensorlabdata1/modelzoo/mlperf/files/image_classification/resnet50_v1_tf.tflite'
input_arrays = ["input_tensor"]
output_arrays = ["resnet_model/dense/BiasAdd"]
input_shapes = {input_arrays[0]: [1, 224, 224, 3]}

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes=input_shapes)
tflite_model = converter.convert()
with open(output_file, "wb") as fp:
    fp.write(tflite_model)
```


<hr>

## MobileNet SSD Object Detection Model
#### Model links
- http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
- https://zenodo.org/record/3163026/files/ssd_mobilenet_v1_coco_2018_01_28.onnx

#### Training Repository
- [tensorflow 1.0 object detection api] https://github.com/tensorflow/models/tree/r1.13.0/research/object_detection
- [tensorflow 1.0 detection model zoo](https://github.com/tensorflow/models/blob/r1.13.0/research/object_detection/g3doc/detection_model_zoo.md)


#### Convert the model to use in TIDL
It may be possible to use the ONNX format of this model as well in TIDL - but we have nto attempted that (That will need an additional prototxt file that describes the Detection Output layer). 

Here we explain how the model can be converted to TFLite format to use in TIDL.

The model can be converted to TFLite format by using the [tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection). 
(Note: There is no need to build tensorflow from source to use the tensorflow object detection api - latest 1.x branch of tensorflow - for example 1.15 - can be used).

Please clone the tensorflow object detection api and switch to the latest 1.x branch. Install the dependencies to use that repository - follow documentation for proper installation.

We start with a checkpoint and get a TensorFlow frozen graph with compatible ops that we can use with TensorFlow Lite. 

To get the frozen graph, run the export_tflite_ssd_graph.py script from the models/research directory with this command: 

Define MODEL_LOCATION environment variable to the path of the model and run the conversion script.

```	
export MODEL_LOCATION=/data/tensorlabdata1/modelzoo/mlperf/files/object_detection/ssd_mobilenet_v1_coco_2018_01_28
export CONFIG_FILE=${MODEL_LOCATION}/pipeline.config
export CHECKPOINT_PATH=${MODEL_LOCATION}/model.ckpt
export OUTPUT_DIR=${MODEL_LOCATION}/

python3 object_detection/export_tflite_ssd_graph.py --pipeline_config_path=$CONFIG_FILE --trained_checkpoint_prefix=$CHECKPOINT_PATH --output_directory=$OUTPUT_DIR --add_postprocessing_op=true --max_detections=4000
```

In the output directory, you should now see two files: tflite_graph.pb and tflite_graph.pbtxt. Note that the add_postprocessing flag enables the model to take advantage of a custom optimized detection post-processing operation which can be thought of as a replacement for tf.image.non_max_suppression. 

Make sure not to confuse export_tflite_ssd_graph with export_inference_graph in the same directory. Both scripts output frozen graphs: export_tflite_ssd_graph will output the frozen graph that we can input to TensorFlow Lite directly and is the one we should be using.


#### Optimize the TFLite model
We will convert the resulting frozen graph (tflite_graph.pb) to the TensorFlow Lite flatbuffer format by running the following python script:

```
graph_def_file = '/data/tensorlabdata1/modelzoo/tflite/object_detection/coco/tensorflow_detection_zoo/others/ssd_mobilenet_v1_coco_2018_01_28/tflite_graph.pb'
output_file = '/data/tensorlabdata1/modelzoo/tflite/object_detection/coco/tensorflow_detection_zoo/others/ssd_mobilenet_v1_coco_2018_01_28/converted_model.tflite'
input_arrays = ['normalized_input_image_tensor']
output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
input_tensor={"normalized_input_image_tensor":[1,300,300,3]}

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays,output_arrays, input_tensor)
converter.allow_custom_ops = True
# converter.inference_type = tf.uint8
# converter.quantized_input_stats = {"normalized_input_image_tensor":(128.0,128.0)}
converter.change_concat_input_ranges = False
tflite_model = converter.convert()
open(output_file, "wb").write(tflite_model)
```

#### Requirements for the converter
- tensorflow=1.15
- tensorflow object detection api that works with tensorflow 1.15

#### Important Links: 
- Tensorflow detection model zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
- https://github.com/tensorflow/tensorflow/issues/28690 
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
- https://stackoverflow.com/questions/58052869/tf-lite-object-detection-only-returning-10-detections
- https://github.com/tensorflow/models/issues/6477  (TFLite low detection accuracy)
- Exporting a trained model for inference: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md
- https://www.tensorflow.org/install/source


<hr>

## ResNet34 SSD Object Detection Model
#### Model links
- https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip?download=1
- https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

#### Training repository
- [mlperf/training/tree/master/single_stage_detector](https://github.com/mlperf/training/tree/master/single_stage_detector)
- [mlperf single_stage_detector inference script](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector/pytorch)

#### How to use in TIDL
The ONNX model can be used in TIDL. And additional prototxt file that describes the Detection Output layer is needed. It is provided in our repository and the usage can be seen in the MLPerf script.
