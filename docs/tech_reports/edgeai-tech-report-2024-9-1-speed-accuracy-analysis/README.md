# Speed/accuracy analysis of object detection models on embedded system

### Authors 
Rekib Zaman, Manu Mathew, Parakh Agarwal <br>
*Processors, Embedded Processing, Texas Instruments*

### Abstract
Low power embedded systems are highly optimized in terms of memory and compute capabilities. Running Deep Neural Network (DNN) models efficiently on these systems is a challenge that needs expertise and tuning for the specific scenario. We propose modifications to popular object detection models that can help them run much faster on embedded systems. We train the models after the modifications and show that the accuracy change is small, while improving the speed significantly.  

Object detection models require several parameters such as top_k, detection_threshold etc. to be tuned carefully to achieve the best speed/accuracy tradeoff. We show the tradeoffs involved in the tuning process. Finally suggestions are provided on parameters suited for various scenarios, such as accuracy measurement or embedded inference.  

### Introduction
DNNs are increasing in complexity and the variety of layer types being used with each passing year. Those who are trying to run their favourite models on a certain embedded system may find that their preferred layer type may not run be running at the expected speed on that particular system. The user then has to request to the Software vendor to improve the support for that layer type - and it may take some time for that support to arrive. And this is not always mechanism for all users. 

#### Model Surgery as an alternative approach
An alternative is to replace the un-optimal layers with equivalent or similar layers.  The replacement should be such that it doesn't bring in signficnat degradation in accuracy after the user finetunes or retrains the model. We call this operation as Model Surgery to create "Lite" models. 

In this study, we provide several such Model Surgery examples - layers that can be replaced in the place of other more complex layers. Several example models were such surgery can be applied are shown. Accuracies are compared before after the Model Surgery & re-training to show that the degradation in accuracy is small.

#### Tradeoffs in speed vs accuracy
TODO

### Models being analyzed
 These are the models being analyzed in this study. The exact configurations used for these experiments are in the configs folder of [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-benchmark).

| **model id** | **model name**      |
|--------------|---------------------|
| od-8020      | ssd_mobilenetv2     |
| od-2060      | ssdlite_mobilenetv2 |
| od-8040      | ssd_regnetx_200mf   |
| od-8880      | yolov8\_small       |
| od-8850      | yolov7\_tiny        |
| od-8860      | yolov7\_large       |
| od-8890      | yolox\_tiny         |
| od-8900      | yolox\_small        |
| od-8800      | yolov4\_csp         |
| od-8820      | yolov5\_small       |
|              |                     |

### **Conversion to Lite model - accuracy impact**

The layers that are not supported in  TIDL are replaced with similar layers that are supported in TIDL. This makes the model embedded friendly but with a small loss in accuracy. Here is a comparison of few models which show how much accuracy drops we get in this process. This accuracy is obtained from running test on COCO dataset in PC. The models that are compared here are yolov8-small, yolov7-large, yolov5-small, yolov4-csp, yolox-tiny, yolov7-tiny and yolovx\_small.  Among these models the highest drop that we see is in yolov5-small, in terms of drop in percentage. we see 5.84% drop in accuracy from 37.7 mAP to 35.5 mAP, i.e. 2.2 mAP drop in lite version. whereas the lowest drop we can see in term of percentage change is in Yolov7\_tiny, where we see a drop of 2.13%  from 37.5 mAP to 36.6 mAP, i.e a drop of 0.8 mAP. So, we see a drop of 2%-6% in lite models.

![](assets/image002.png) 

|     |     |     |
| --- | --- | --- |
| models\_name | non-lite-model-acc | lite-model-acc |
| yolov8\_small | 44.2 | 42.4 |
| yolov7\_large | 51  | 48.1 |
| yolov5\_small | 37.7 | 35.5 |
| yolov4\_csp | 48  | 45.8 |
| yolox\_tiny | 32.7 | 31.1 |
| yolov7\_tiny | 37.5 | 36.7 |
| yolox\_small | 40.7 | 38.7 |
|   |   |

### **Changes in Lite models :** 

##### ·          **SiLU, LeakyReLU, Mish  to ReLU :** 

The activation functions like SiLU, LeakyReLU, Mish are not supported in TIDL so we replace it ReLU activation function in the lite model

|     |     |
| --- | --- |
| SiLU, LeakyReLU, Mish                         | ReLU                           |
| ![](assets/image004.png)  |  ![](assets/image006.png)  |
| | |

*   **MaxPool 5x5,9x9,13x13 to MaxPool 3x3**

MaxPool layers with kernel size 5x5, 9x9, 13x13 are not supported in TIDL so they are replaced with MaxPool of kernel size 3x3 . Two consecutive MaxPool layer of kernel size 3x3 gives same output as one Maxpool layer of kernel size 5x5. So 1 maxpool of size 5x5 is replaced with 2 maxpool of size 3x3. Similarly maxpool of size 9x9 and 13x3 is replaced with 4 maxpool layer of size 3x3 and 6 maxpool layer of size 3x3 respectively. But point to be noted is that, as it does not involve ant weights , this change does not affect in accuracy.

|     |     |
| --- | --- |
| MaxPool 5x5,9x9,13x13                         | MaxPool 3x3                        |
| ![](assets/image008.png) |  ![](assets/image009.png) |
| | |



##### ·         **Focus Layer to Conv layer :** 

As Focus layer involves data movement, it is not fully optimized for TIDL. Focus layer reduced the input size to half, so we perform this using conv layer of size 5x5 with hardware accelerated conv operation.

|     |     |
| --- | --- |
|  Focus Layer                         | replaced Conv 5x5                      |
| ![](assets/image010.png)   |  ![](assets/image011.png) |
| | |

                                

                                                                                                                                       

*   **Conv 6x6 to Conv 5x5**

Convolution layer of kernel size 6x6 in yolov5 has been replaced with convolution layer of kernel size 5x5, as TIDL does not support 6x6 conv layer as of now.

### List of changes in each models : 

|     |     |     |
| --- | --- | --- |
| **Model names** | **original models** | **lite models** |
| yolov8\_small | SilU<br><br>MaxPool 5x5 | ReLU<br><br>2 MaxPool 3x3 |
| yolov4\_csp | Mish<br><br>MaxPool 5x5<br><br>MaxPool 9x9<br><br>MaxPool 13x13 | ReLU<br><br>2 MaxPool 3x3<br><br>4 MaxPool 3x3<br><br>6 MaxPool 3x3 |
| yolov5\_small| Conv 6x6<br><br>SiLU<br><br>MaxPool 5x5 | Conv 5x5<br><br>ReLU<br><br>2 MaxPool 3x3 |
| yolov7\_tiny , <br>yolov7\_large | LeakyReLU<br><br>MaxPool 5x5 | ReLU<br><br>2 MaxPool 3x3 |
| yolox\_tiny ,  <br>yolox\_small | SiLU<br><br>Focus Layer<br><br>MaxPool 5x5<br><br>MaxPool 9x9<br><br>MaxPool 13x13 | ReLU<br><br>Conv Layer 5x5<br><br>2 MaxPool 3x3<br><br>4 MaxPool 3x3<br><br>6 MaxPool 3x3 |


## **Effect of mixed-precision**

We see a drop in accuracy when we run the full model in 8bit mode. We see up to 7%mAP drop in some models when ran on pure 8bit mode, But the drop was recovered when we run the models in mixed-precision mode.  In mixed-precision we basically run the end layers of the neck module in 16bit mode, which helps to recover the drop in 8bit.

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| models\_name | non-lite-model-acc | lite-model-acc | 32bit-acc | 8bit-acc | 8bit-acc-mp | GMACs |
| yolov8\_small | 44.2 | 42.4 | 41.44 | 39.881021 | 40.702836 | 14.3337 |
| yolov7\_large | 51  | 48.1 | 46.93 | 42.242723 | 46.63116 | 52.9457 |
| yolov5\_small | 37.7 | 35.5 | 34.83 | 31.953114 | 34.108788 | 8.1323 |
| yolov4\_csp | 48  | 45.8 | 45.3 | 42.443044 | 45.135142 | 60.1735 |
| yolox\_tiny | 32.7 | 31.1 | 30.56 | 23.157898 | 30.445955 | 3.2534 |
| yolov7\_tiny | 37.5 | 36.7 | 35.75 | 32.146389 | 35.50527 | 6.8741 |
| yolox\_small | 40.7 | 38.7 | 38.11 | 32.346607 | 37.921559 | 13.4596 |
|     |     |     |     |     |     |     |

![](assets/image013.png)

![](assets/image015.png)

![](assets/image017.png)

## **Performance on TIDL**

In this section we compare the performance of the models running in TIDL. For this experiment we use four YOLO models, yolov7-large, yolov5-small, yolov4-csp and yolov7-tiny . We first run the whole model without TIDL i.e, fully in ARM mode. Then we run only the backbone(backbone+neck) part in TIDL and the head part in ARM mode. Then we finally run the full model in TIDL. We also run the models one more time in TIDL by adjusting the detection threshold to get better accuracy. We initially used detection threshold of 0.25 and topK of 100, then we reduced the detection threshold to 0.05 and increased the topK to 500. This table shows the accuracy as well as inference time :

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
|     |     | yolov7\_large | yolov5\_small | yolov4\_csp | yolov7\_tiny |
| **Accuracy\[.5:.95\]%** | **without\_tidl** | 42.126174 | 30.196424 | 40.028126 | 30.724629 |
| **Accuracy\[.5:.95\]%**    | **backbone\_tidl** | 41.588899 | 29.270204 | 38.1228 | 29.842514 |
| **Accuracy\[.5:.95\]%**    | **tidl\_25\_100** | 42.211561 | 29.768259 | 40.065406 | 30.702776 |
| **Accuracy\[.5:.95\]%**    | **tidl\_05\_500** | 45.711503 | 33.412331 | 44.316673 | 34.626288 |
|     |     |     |     |     |     |     |
| **Inference Time** | **without\_tidl** | 6671.132608 | 1088.245636 | 7560.71492 | 956.867193 |
| **Inference Time**    | **backbone\_tidl** | 101.190856 | 65.981589 | 115.274344 | 65.550423 |
| **Inference Time**    | **tidl\_25\_100** | 46.604622 | 13.175834 | 44.581183 | 11.941207 |
| **Inference Time**    | **tidl\_05\_500** | 46.921291 | 13.77424 | 45.188118 | 12.216323 |
|     |     |     |     |     |     |     |




### Effect on Inference time :

We see very high inference time when running the model on ARM mode. The large models like yolov7-large and yolov4 takes around 7 seconds for a single frame. When we run the backbone in TIDL we see drastic change in inference time. Yolov7-large which took 6671ms for 1 frame in ARM mode tool only 101ms by running the backbone in TIDL.

![](assets/image019.png)

Now, when we run the full model in TIDL we see further improvement in inference time. The inference time for yolov7-large further reduced from 101ms to 46ms. The below table shows the fps for the models on running on TIDL

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|     | yolov7\_large | yolov5\_small | yolov4\_csp | yolov7\_tiny |
| fps | 21.45709926 | 75.89652389 | 22.4309884 | 83.74362826 |
|     |     |     |     |     |

When we further adjust the detection threshold for better accuracy we see negligible change in inference time. So we can use the most suitable detection threshold for each model to get maximum accuracy with minimum change in inference time. We will discuss threshold adjustment in further section.

### Effect on Accuracy :

As expected we don’t see much change in accuracy when we run model in ARM mode or in TIDL. We see improvement in accuracy when we adjust the detection threshold also without any change in inference time as we have seen in previous section.

![](assets/image021.png)

## **Tuning detection threshold for better accuracy**

As we saw in the previous section that reducing the detection threshold can improve the accuracy, but if we reduce the threshold by huge amount the accuracy might get better but it will worsen the performance. So what is the right detection threshold to get better accuracy with  minimum change in the inference time. We perform an experiment to get the idea about which detection threshold and topK works better for which detection models. In this experiment we use 5 YOLO and 3 SSD od-models. We use 3 different set of confidence threshold of 0.3, 0.05 and 0.01 in combination with three different topK of 200, 500 and 1000 to check how the accuracy metric and inference time varies while tuning those values, also to notice the trade-off between accuracy and inference time.

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
|     | yolov7\_tiny |     |     | ssd\_regnetx\_200mf\_320 |     |     |
|     | Inference Time (ms) | 8bits Accuracy | float Accuracy | Inference Time (ms) | 8bits Accuracy | float Accuracy |
| ConfidenceTh=0.3 Topk=200 | 11.92091 | 29.82639 | 30.538 | 6.10956 | 15.602666 | 17.33 |
| ConfidenceTh=0.05 Topk=200 | 12.074695 | 34.645118 |     | 12.071568 | 16.950246 |     |
| ConfidenceTh=0.05 Topk=500 | 12.103096 | 34.688628 |     | 16.96855 | 16.914372 |     |
| ConfidenceTh=0.05 Topk=1000 | 12.106057 | 34.689615 |     | 24.56903 | 16.914489 |     |
| ConfidenceTh=0.01 Topk=200 | 12.685948 | 35.45477 |     | 41.056791 | 16.974615 |     |
| ConfidenceTh=0.01 Topk=500 | 12.902015 | 35.534706 |     | 77.539697 | 16.938978 |     |
| ConfidenceTh=0.01 Topk=1000 | 13.016959 | 35.542307 | 35.75 | 149.33048 | 16.936272 | 20.63 |
|     |     |     |     |     |     |     |

Here in this table we compare two models, one YOLO model and one SSD model. We observe that on reducing the confidence threshold we see improvement in accuracy and some increment in inference time as well. For yolov7-tiny model, we see a jump in accuracy by almost 5% mAP when reducing the confidence threshold from 0.3 to 0.05, which is a huge difference. Also further reducing it to 0.01 we get almost 1% mAP improvement in accuracy. When see the inference time for this model we don't see any big increment if not zero. Whereas for the SSD model we don't see any huge improvement in accuracy but we can notice that the inference time is gradually getting worse with reduction of confidence threshold and increment of topK . The figure below shows the comparison of all the 8 models. some of the major observations are : 

![](assets/image023.png)

*   We see significant change in inference times in SSD models on reducing the detection threshold, while improvement in accuracy is almost zero.
*   SSD model shows improvement in accuracy when reducing the detection threshold from 0.3 to 0.05 but at the cost of little increase in inference time. And with further reduce in detection threshold we don't see any improvement in accuracy.
*   For YOLO models we see huge improvement in accuracy in all the models on reducing the detection threshold from 0.3 to 0.05 also with negligible change in inference time.
*   We see further improvements in accuracy in YOLO models on further reducing the detection threshold from 0.05 to 0.01, We also see some increase in inference time but it is negligible.
*   Whereas in case of SSD models, we don't see any improvements in accuracy, but the inference time is increased by huge amount.

So from this  experiment we can conclude that : 

*   For YOLO models, we can reduce the detection threshold as it does not effect the inference time significantly while improving the accuracy
*   For SSD models, higher threshold is preferable, but for little improvement in accuracy we can reduce the detection threshold to 0.05 which will also increase the inference time.
*   So there is a trade-off between accuracy and inference time for SSD models. We can use detection threshold of 0.3 for better fps and 0.05 for better accuracy.