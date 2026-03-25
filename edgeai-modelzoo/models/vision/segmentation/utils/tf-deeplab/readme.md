### Tensorflow Deeplab Models
- [Models Source Code](https://github.com/tensorflow/models/tree/master/research/deeplab)

## Model Details
* Tensorflow Deeplab has scripts and pre-trained models for several popular model architectures. Several datasets such as Cityscapes, Pascal VOC and ADE20K are supported.<br>

### Graph transformation 
* We have slightly modified these models for faster execution.<br>
* Following changes were made from the existing frozen graph without any further training. These changes resulted in slight degradation in accuracy.<br>
    + Input resolution changed to power of two. (E.g. 513x513 is changed to 512x512).
    + Resize factor changed to power of two. (All different resizing are made pow of 2).
    + Avg-pooling changed to power of two. (E.g 65x65 is changed to 64x64).
    + In these models, depthwise convolutions with dilation comes as {SpaceToBatch, depthwise convolutions, BatchToSpace} layers. To accommodate above changes, modifications were made in these layers. 

* In order to make the above changes, we have provided a script in [tf_deeplab_frozen_graph_transforms.py](./tf_deeplab_frozen_graph_transforms.py). Follow the steps below to run the script. The script uses graph transformation tool that was readily available in TF 1.XX.

    ```
    pip install tensorflow==1.15 
    ```
    ```
    wget http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz
    tar xvf deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz
    python  frozen_graph_transforms.py --pb_path ./deeplabv3_mnv2_dm05_pascal_trainaug/frozen_inference_graph.pb --input_nodes sub_7 --output_nodes ArgMax --input_shape 512 512
    ```

*   The above script has been validated for several models.

### Results

- [Training Source Code](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [**Additional Information**](./utils/README_tf_deeplab.md) We have slightly modified these models for faster execution and details are in this link.

|Dataset    |Model Name                     |Input Size |GigaMACs  |MeanIoU%    |
|---------- |------------------------------ |-----------|----------|------------|
|-          |**Our modified models**
|VOC2012    |deeplabv3_mnv2_dm05            |512x512    |2.77      |66.94       |
|VOC2012    |deeplabv3_mnv2                 |512x512    |8.60      |72.66       |
|VOC2012    |deeplabv3_xception             |512x512    |171.77    |81.74       |
|-          |**Original DeepLab models**
|Cityscapes |MobileNetV2+DeepLab            |769x769     |21.27     |70.71       |
|Cityscapes |MobileNetV3+DeepLab            |769x769     |15.95     |72.41       |
|Cityscapes |Xception65+DeepLab             |769x769     |418.64    |78.79       |





