# Training for Image Classification
 
Image Classification is a fundamental task in Deep Learning and Computer Vision. Here we show couple of examples of training CNN / Deep Learning models for Image Classification. For this example, we  use MobileNetV2 as the model for training, but other models can also be used.

Commonly used Traning/Validation commands are listed in the file [run_edgeailite_classification.sh](../../run_edgeailite_classification.sh). Uncommend one and run the file to start the run. 

<br><hr><br>


## Cifar Dataset 
Cifar10 and Cifar100 are popular Datasets used for training CNNs. Since these datasets are small, the training can be finished in a a short time and can give an indication of how good a particular CNN model is. The images in these datasets are small (32x32).

### Cifar100 Dataset
Since the dataset is small, the training script itself can download the dataset before training.

Training can be started by the following command:<br>
```
python ./references/classification/train_classification_main.py --dataset_name cifar100_classification --model_name mobilenetv2_tv_x1 --data_path ./data/datasets/cifar100_classification --img_resize 32 --img_crop 32 --rand_scale 0.5 1.0
```
 
In the script, note that there are some special settings for the cifar datasets. The most important one is the 'strides' settings. Since the input images in Cifar are small we do not want to have as many strides as in a large sized image. See the argument args.model_config.strides being set in the script.

During the training, **validation** accuracy will also be printed. But if you want to explicitly check the accuracy again with **validation** set, it can be done:<br>
```
python ./references/classification/train_classification_main.py --phase validation --dataset_name cifar100_classification --model_name mobilenetv2_tv_x1 --data_path ./data/datasets/cifar100_classification --img_resize 32 --img_crop 32
```

### Cifar10 Dataset
 Training can be started by the following command:<br>
```
python ./references/classification/train_classification_main.py --dataset_name cifar10_classification --model_name mobilenetv2_tv_x1 --data_path ./data/datasets/cifar10_classification --img_resize 32 --img_crop 32 --rand_scale 0.5 1.0
```

<br><hr><br>


## ImageNet Dataset
It is difficult to reproduce the accuracies reported in the original papers for certain models (especially true for MobileNetV1 and MobileNetV2 models) due to the need for careful hyper-parameter tuning. In our examples, we have incorporated hyper-parameters required to train high accuracy classification models.

Important note: ImageNet dataset is huge and download may take long time. Attempt this only if you have a good internet connection. Also the training takes a long time. In our case, using four GTX 1080 Ti, it takes nearly two days to train.

Training can be started by the following command:<br>
```
python ./references/classification/train_classification_main.py --dataset_name imagenet_classification --model_name mobilenetv2_tv_x1 --data_path ./data/datasets/imagenet_classification
```

Training with ResNet50:<br>
```
python ./references/classification/train_classification_main.py --dataset_name imagenet_classification --model_name resnet50_x1 --data_path ./data/datasets/imagenet_classification
```
  
After the training, the **validation** accuracy using (make sure that  args.dataset_name and args.pretrained are correctly set)<br>
```
python ./references/classification/train_classification_main.py --phase validation --dataset_name imagenet_classification --model_name mobilenetv2_tv_x1 --data_path ./data/datasets/imagenet_classification --pretrained <checkpoint_path>
```

## ImageNet or any other classification dataset - manual download
In this case, the images of the dataset is assumed to be arranges in folders. 'train'  and 'validation' are two separate folders and underneath that, each class should have a different folder.

Assume that that folder './data/datasets/image_folder_classification' has the  the classification dataset. This folder should contain folders and images as follows: 
  image_folder_classification<br>
  &nbsp;&nbsp;train<br>
  &nbsp;&nbsp;&nbsp;&nbsp;class1<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image files here<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;....<br>
  &nbsp;&nbsp;&nbsp;&nbsp;class2<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image files here<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;....<br>
  &nbsp;&nbsp;validation<br>
  &nbsp;&nbsp;&nbsp;&nbsp;class1<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image files here<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;....<br>
  &nbsp;&nbsp;&nbsp;&nbsp;class2<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image files here<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;....<br>

Note 'class1', 'class2' etc are examples and they stand for the names of the classes that we are trying to classify.

Here we use ImageNEt dataset as an example, but it could ne any other image classification dataset arranged in folders. 

The download links for ImageNet are given in [torchvision/datasets/imagenet.py](../../torchvision/datasets/imagenet.py)
```
cd ./data/datasets/image_folder_classification
# download the imagenet tar files and then execute the following
mkdir train
mkdir validation
tar -C train -xvf ILSVRC2012_img_train.tar
tar -C validation -xvf ILSVRC2012_img_val.tar
```

After downloading and extracting, use this script to arrange the validation folder into folders of classes: 
```
cd validation
wget https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh
./valprep.sh
rm ./valprep.sh
```

Training with **MobileNetV2** model can be started by the following command from the base folder of the repository:<br>
```
python ./references/classification/train_classification_main.py --dataset_name image_folder_classification --model_name mobilenetv2_tv_x1 --data_path ./data/datasets/image_folder_classification
```

Training with **ResNet50** model:<br>
```
python ./references/classification/train_classification_main.py --dataset_name image_folder_classification --model_name resnet50_x1 --data_path ./data/datasets/image_folder_classification
```

Training with **RegNet800MF model and BGR image input transform**:<br>
```python ./references/classification/train_classification_main.py --dataset_name image_folder_classification --model_name regnetx800mf_x1 --data_path ./data/datasets/image_folder_classification --input_channel_reverse True --image_mean 103.53 116.28 123.675 --image_scale 0.017429 0.017507 0.017125
```

If the dataset is in a different location, it can be specified by the --data_path option, but dataset_name must be *image_folder_classification* for folder based classification.


<br><hr><br>


## Results

### ImageNet (ILSVRC2012) Classification (1000 class)

ImageNet classification results are as follows:

|Dataset  |Mode Name          |Resize Resolution|Crop Resolution|Complexity (GigaMACS)|Top1 Accuracy% |Model Configuration Name|
|---------|----------         |-----------      |----------     |--------             |--------       |------------------------|
|ImageNet |MobileNetV1        |256x256          |224x224        |0.568                |**71.83**      |mobilenetv1_x1          |
|ImageNet |MobileNetV2        |256x256          |224x224        |0.296                |**72.13**      |mobilenetv2_tv_x1       |
|ImageNet |ResNet50-0.5       |256x256          |224x224        |1.051                |**72.05**      |resnet50_xp5            |
|ImageNet |**RegNetX800MF**   |256x256          |224x224        |0.800                |               |regnetx800mf_x1         |
|.
|ImageNet |MobileNetV2-QAT*   |256x256          |224x224        |0.296                |71.76          |                        |
|.
|ImageNet |MobileNetV1[1]     |256x256          |224x224        |0.569                |70.60          |                        |
|ImageNet |MobileNetV2[2]     |256x256          |224x224        |0.300                |72.00          |                        |
|ImageNet |ResNet50[3]        |256x256          |224x224        |4.087                |76.15          |                        |
|ImageNet |**RegNet4800MF**[4]|256x256          |224x224        |0.400                |**72.6**       |                        |
|ImageNet |**RegNetX800MF**[4]|256x256          |224x224        |0.800                |**75.2**       |                        |
|ImageNet |RegNetX1.6F[4]     |256x256          |224x224        |1.6                  |**77.0**       |                        |

*- Quantization Aware Training using 8b precision


<br><hr><br>


#### Notes
- As can be seen from the table, the models included in this repository provide a good Accuracy/Complexity tradeoff. 
- However, the Complexity (in GigaMACS) does not always indicate the speed of inference on an embedded device. We have to also consider the fact that regular convolutions and Grouped convolutions are typically more efficient in utilizing the available compute resources (as they have more compute per data trasnsfer) compared to Depthwise convolutions.
- Overall, RegNetX based models strike a good balance between complexity, accuracy and easiness of quantization.


<br><hr><br>


## Referrences

[1] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, Andreetto M, Adam H, arXiv:1704.04861, 2017

[2] MobileNetV2: Inverted Residuals and Linear Bottlenecks, Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC. arXiv preprint. arXiv:1801.04381, 2018.

[3] PyTorch TorchVision Model Zoo: https://pytorch.org/docs/stable/torchvision/models.html

[4] Designing Network Design Spaces, Ilija Radosavovic Raj Prateek Kosaraju Ross Girshick Kaiming He Piotr Dollar´, Facebook AI Research (FAIR), https://arxiv.org/pdf/2003.13678.pdf, https://github.com/facebookresearch/pycls
