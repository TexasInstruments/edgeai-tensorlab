
downloads_folder=./dependencies/downloads

#######################################################
# ImageNet Dataset
echo "To download ImageNet dataset, please visit http://image-net.org/ and request permission."
read -p "After that press [ENTER]:"

mkdir -p $imagenet_folder

imagenet_folder=$downloads_folder/imagenet

imagenet_tar=$downloads_folder/ILSVRC2012_img_val.tar
wget --show-progress http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar -O $imagenet_tar
tar -xf $imagenet_tar -C $imagenet_folder/val

wget --show-progress http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz -O $downloads_folder/caffe_ilsvrc12.tar.gz
tar -xf $downloads_folder/caffe_ilsvrc12.tar.gz -C $imagenet_folder

#######################################################
# COCO Dataset
echo "Downloading the COCO dataset"
mkdir -p $downloads_folder/coco/val2017
mkdir -p $downloads_folder/coco/annotations

wget --show-progress http://images.cocodataset.org/zips/val2017.zip -O $downloads_folder/val2017.zip
unzip -q $downloads_folder/val2017.zip -d $downloads_folder/coco/val2017

wget --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $downloads_folder/annotations_trainval2017.zip
unzip -q $downloads_folder/annotations_trainval2017.zip -d $downloads_folder/coco/annotations

#######################################################
# ADE20K Dataset
echo "Downloading the ADE20K dataset"
mkdir -p $downloads_folder/ADEChallengeData2016

wget --show-progress http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O $downloads_folder/ADEChallengeData2016.zip
unzip -q $downloads_folder/ADEChallengeData2016.zip -d $downloads_folder/ADEChallengeData2016

#######################################################
# Pascal VOC 2012 dataset
mkdir -p $downloads_folder/VOCdevkit/VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar -o $downloads_folder/VOCdevkit_18-May-2011.tar
tar -xf $downloads_folder/VOCdevkit_18-May-2011.tar -C $downloads_folder/VOCdevkit
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -o $downloads_folder/VOCtrainval_11-May-2012.tar
tar -xf $downloads_folder/VOCtrainval_11-May-2012.tar -C $downloads_folder/VOCdevkit/VOC2012
## alternate URL - this is untested
##wget --show-progress https://data.deepai.org/PascalVOC2012.zip -o $downloads_folder/PascalVOC2012.zip
##unzip -q $downloads_folder/PascalVOC2012.zip -d $downloads_folder/VOCdevkit
python $downloads_folder/remove_gt_colormap_voc.py \
      --original_gt_folder "${downloads_folder}/VOCdevkit/VOC2012/SegmentationClass" \
      --output_dir "${downloads_folder}/VOCdevkit/VOC2012/SegmentationClassRaw"
