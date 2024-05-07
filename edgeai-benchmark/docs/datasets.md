
## ImageNet dataset 
Since the ImageNet download URL is not currently working, one need to download this dataset manually. If you download and place the tar file ILSVRC2012_img_val.tar in the folder dependencies/datasets/imagenet our script can take care of further processing - so the steps after that download is for information only if you wish to do that manually.

ImageNet dataset can be obtained by registering at their website. Register at http://www.image-net.org/

In this repository, we are only interested in the validation split of ImageNet. ImageNet dataset can be made available a the path dependencies/datasets/imagenet. The procedure for doing that is as follows:

After login, click on 2012 in the section Download links to ILSVRC image data.  Inside that you can see the link for the file ILSVRC2012_img_val.tar:

Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622

Download this tar file into the folder dependencies/datasets/imagenet. 

Once the download is complete, untar the contents into the folder *dependencies/datasets/imagenet/val*. It should have exactly 50,000 images.

Now download the file http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

Untar inside the folder dependencies/datasets/imagenet. After that folder *dependencies/datasets/imagenet* should have a file called val.txt

After these operations, we should have the folder *dependencies/datasets/imagenet/val* with 50,000 images and and a text file  *dependencies/datasets/imagenet/val.txt*

## COCO dataset
Our script can download COCO dataset automatically - so the following is for information only.

COCO dataset should be available in the path *dependencies/datasets/coco* 

We are interested in the validation split in the folders *dependencies/datasets/coco/val2017* and *dependencies/datasets/coco/annotations*

The links for downloading these are available in https://cocodataset.org/#download

## ADE20K dataset
Our script can download ADE20K dataset automatically - so the following is for information only.

ADE20K dataset should be available in the path *dependencies/datasets/ADEChallengeData2016*

Information about this dataset is available in https://groups.csail.mit.edu/vision/datasets/ADE20K

It can be downloaded from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
        
## PascalVOC2012 dataset
Our script can download PascalVOC2012 dataset automatically - so the following is for information only.

PascalVOC2012 dataset should be available in the path *dependencies/datasets/VOCdevkit/VOC2012*

Information about this dataset can be obtained from the URL http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

This dataset needs two tar files which can be downloaded from: 

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar

If you download and place these tar files in the folder dependencies/datasets/VOCdevkit/VOC2012 our script can take care of further processing.

After processing the follwoing folders must be available: *dependencies/datasets/VOCdevkit/VOC2012/Annotations*, *dependencies/datasets/VOCdevkit/VOC2012/ImageSets*, *dependencies/datasets/VOCdevkit/VOC2012/JPEGImages*, *dependencies/datasets/VOCdevkit/VOC2012/SegmentationClass*, *dependencies/datasets/VOCdevkit/VOC2012/SegmentationClassRaw*

## Cityscapes dataset
Cityscapes dataset  is not freely available for download, but you can obtain it for experimentation purposes by registering in their website https://www.cityscapes-dataset.com/.

The dataset should be available in the path *dependencies/datasets/cityscpaes* - we use only the validation split - which should be in the folders *dependencies/datasets/cityscpaes/cityscapes/leftImg8bit/val* and *dependencies/datasets/cityscpaes/cityscapes/gtFine/val*<br>

## KITTI dataset
KITTI dataset downloading needs user registration hence our script can not download it automatically. User is requested to register at
https://www.cvlibs.net/datasets/kitti/ and proceed with manual downloading of the dataset. KITTI dataset is needed for 3D object detection benchmarking.

KITTI 3D-OD dataset needs preprocessing before it can be used in this script for benchmarking. User can refer https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d#dataset-preperation for downloading and preprocessing the dataset for 3D-OD benchmarking task.   

