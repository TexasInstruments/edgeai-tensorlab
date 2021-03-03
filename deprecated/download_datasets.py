import os
from jacinto_ai_benchmark import *

print("NOTICE: this script is deprecated. Please use configs.download_datasets(settings)")
os.exit()

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

imagenet = datasets.ImageNetCls(path='./dependencies/downloads/imagenet/val', split='val.txt', download=True)
coco = datasets.COCODetection(path='./dependencies/downloads/coco', split='val2017', download=True)
ade20k = datasets.ADE20KSegmentation(path='./dependencies/datasets/ADEChallengeData2016', split='validation', download=True)
voc2012 = datasets.VOC2012Segmentation(path='./dependencies/datasets/VOCdevkit/VOC2012', split='trainaug', download=True)