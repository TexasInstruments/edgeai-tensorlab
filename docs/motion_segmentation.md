# Motion Segmentation
Motion segmentation network predicts the state of motion for each pixel. 

The model used for this task uses two stream architecture as shown in the figure below . Two parallel encoders extract appearance and temporal features separately and fuse them at stride of 4. 

<p float="left">
  <img src="motion_segmentation/motion_segmentation_network.PNG" width="555" hspace="5"/>
</p>

## Models
We provide  scripts for training models with three different input combinations. They are image pair and (optical flow, current image) and (opticalflow with confidence, current image) respectively. Optical flow is generated using current frame and previous frame.

The following model is used for training:

**deeplabv3plus_mobilenetv2_tv_edgeailite**: This is same as deeplabv3plus_mobilenetv2_tv_edgeailite as described [`here`](Semantic_Segmentation.md). The sole difference being it takes two inputs for the training and fuses the feature maps after a stride of 4. This increases the complexity compared to a single stream model by roughly 20%.

## Datasets: Cityscapes Motion Datset
**Dataset preparation:**  Dataset used for this training is cityscapes dataset with motion annotation. This training requires either the previous frame or optical flow generated from (current frame, previous frame). Given below are the details to download these extra files.

* Clone this [`repository`](https://bitbucket.itg.ti.com/projects/ALGO-DEVKIT/repos/cityscapes_motion_dataset/browse) for all the tools required to proceed further.

* **Current frame:**: This is can be downloaded from https://www.cityscapes-dataset.com/. Download the zip file leftImg8bit_trainvaltest.zip. keep the directory leftimg8bit in ./data/datatsets/cityscapes/data/.

* **Previous frame:** Previous frames can be downloaded from https://www.cityscapes-dataset.com/ as well. The zip file is cityscapes_leftImg8bit_sequence_trainvaltest.zip. The previous frame corresponds to the 18th frame in each non-overlapping 30 frame snippet. Once you have downloaded the entire sequence, run the script `filter_cityscape_previous_frame.py` from ths [`repository`](https://bitbucket.itg.ti.com/projects/ALGO-DEVKIT/repos/cityscapes_motion_dataset/browse)  to extract the previous frame from the sequence. This script will keep the previous frame in a directory named leftImg8bit_previous. Move leftImg8bitPrevious to ./data/datatsets/cityscapes/data/.
 
* **Optical flow:** Optical flow can be generated using current frame and previous frame. This repository contains optical flow generated from (current frame, previous frame). Run **this** script to generate the flow if you have downloaded the previous frame. Otherwise this repository will have two folders named `leftimg8bit_flow_farneback` and `leftimg8bit_flow_farneback_confidence` . Move them to location ./data/datatsets/cityscapes/data/. Optical flow has been generated using opencv farneback flow with some parameter tuning whereas confidence is computed using forward-backward consistency.

* **Motion annotation:**  The same repository as above contains motion annotation as well. They are inside `gtFine`. Move the gtFine directory into ./data/datatsets/cityscapes/data/. Here, moving pixels correspond to 255 whereas static pixels are marked with 0.

Now, the final directory structure must look like this:
```
./data/datatsets/cityscapes/data/
    leftimg8bit	                                                -  Current frame from cityscapes dataset
    leftImg8bit_previous                                    -  Previous frame from cityscapes dataset  
    leftimg8bit_flow_farneback	                    -  Optical flow generated from (Curr_frame, Previous Frame) and stored in the format(u',v',128)
    leftimg8bit_flow_farneback_confidence  -  Optical flow, confidence generated from (Curr_frame, Previous Frame) and stored in the format(u',v',confidence)
    gtFine	                                                -  Ground truth with motion annotation
```

 
 Following are the  commands for training networks for `various inputs`:
<br>

**(Previous frame, Current frame):** 
```
python ./references/pixel2pixel/train_motion_segmentation_main.py --image_folders  leftImg8bit_previous leftImg8bit --is_flow 0,0 0 --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1
```
**(Optical flow, Current frame):**
``` 
python ./references/pixel2pixel/train_motion_segmentation_main.py --image_folders leftImg8bit_flow_farneback leftImg8bit --is_flow 1,0 0 --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1
```
**(Optical flow with confidence, Current frame):**
```  
python ./references/pixel2pixel/train_motion_segmentation_main.py --image_folders leftImg8bit_flow_farneback_confidence leftImg8bit --is_flow 1,0 0 --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1
```

## Results:
## Cityscapes Motion segmentation
| Inputs                        | mIOU (static class, moving class) |
|-----------------------------------------------|-------------------|
| Previous frame, Current frame                  | 83.1 (99.7,66.5) |
| Optical flow, Current frame                    | 84.8 (99.7,69.9) |
| Optical flow with confidence, Current frame    | 85.5 (99.7,71.3) |

Using (optical flow, curr frame) , we get around ~3.4% improvement for moving class over image pair. Using confidence, we get an overall improvement of 4.8% over image_pair baseline. The results above show that, we can achieve significant improvement for motion segmentation using optical flow and it further improves with confidence measure for flow.
## References
[1]The Cityscapes Dataset for Semantic Urban Scene Understanding, Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele, CVPR 2016, https://www.cityscapes-dataset.com/
