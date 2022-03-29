import os
import sys
import json
import glob
import PIL
import PIL.Image
import math


source_dir = '/data/ssd/files/a0393608/work/code/ti/algoref/vision-dataset/annotatedJSON/tiscapes/data/gtFine'
dest_dir = '/data/ssd/files/a0393608/work/code/ti/algoref/vision-dataset/annotatedJSON/tiscapes/data/annotations'

split_names = ['train', 'val']
polygon_suffix = '_gtFine_polygons.json'
source_image_suffix = '.png'
dest_image_suffix = '.jpg'

# the background class is anything except the other four classes
# background class is not fully annotated - so should not be used for training
# it is basically used to indicated an area that overlaps with a fully annotated class, only whenever that happens.
classes_selected = ['background', 'road', 'human', 'roadsign', 'vehicle']

classes_mapping = {
    'background': 'background',
    'unlabeled': 'background',
    'ego vehicle': 'background',
    'rectification border': 'background',
    'out of roi': 'background',
    'static': 'background',
    'dynamic': 'background',
    'ground': 'background',
    'road': 'road',
    'sidewalk': 'background',
    'parking': 'background',
    'rail track': 'background',
    'building': 'background',
    'wall': 'background',
    'fence': 'background',
    'guard rail': 'background',
    'bridge': 'background',
    'tunnel': 'background',
    'pole': 'background',
    'polegroup': 'background',
    'traffic sign': 'roadsign',
    'traffic light': 'roadsign',
    'vegetation': 'background',
    'terrain': 'background',
    'sky': 'background',
    'person': 'human',
    'rider': 'human',
    'car': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'caravan': 'vehicle',
    'trailer': 'vehicle',
    'train': 'vehicle',
    'motorcycle': 'vehicle',
    'bicycle': 'vehicle',
    'license plate': 'background',
}
background_class = 'background'

ann_data = {}
annotations = []
images = []
image_id = 0
annotation_id = 0
category_id_start = 0 #1
categories = [dict(id=class_id+category_id_start, name=class_name) for class_id, class_name in enumerate(classes_selected)]

os.makedirs(dest_dir, exist_ok=True)

for split_name in split_names:
    json_files = glob.glob(f'{source_dir}/{split_name}/*/*{polygon_suffix}')
    for json_file in json_files:
        with open(json_file) as jfp:
            json_data = json.load(jfp)
        #
        image_file = json_file.replace(polygon_suffix, source_image_suffix).replace('gtFine', 'leftImg8bit')
        image = PIL.Image.open(image_file)
        image_size = image.size
        image_file = os.path.basename(image_file)
        file_name = f'{os.path.splitext(image_file)[0]}{dest_image_suffix}'
        image_info = {
            'id': image_id,
            'license': None,
            'width': image_size[0],
            'height': image_size[1],
            'file_name': file_name,
            'split_name': split_name
        }
        images.append(image_info)

        for obj in json_data['objects']:
            deleted = obj['deleted']
            if deleted:
                annotation_id += 1
                continue
            #
            label = obj['label']
            label_mapped = None
            if label in classes_mapping:
                label_mapped = classes_mapping[label]
            else:
                for label_name in classes_mapping:
                    if label_name.startswith(label):
                        label_mapped = classes_mapping[label_name]
                    #
                #
            #
            label_mapped = label_mapped or label

            polygon = obj['polygon']
            segmentation = []
            bbox_xyxy = [math.inf, math.inf, 0, 0]
            for p in polygon:
                segmentation.extend([p[0], p[1]])
                bbox_xyxy[0] = min(bbox_xyxy[0], p[0])
                bbox_xyxy[1] = min(bbox_xyxy[1], p[1])
                bbox_xyxy[2] = max(bbox_xyxy[2], p[0])
                bbox_xyxy[3] = max(bbox_xyxy[3], p[1])
            #
            segmentation = [round(s, 2) for s in segmentation]
            if label_mapped != background_class:
                bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
                bbox_xywh = [round(v, 2) for v in bbox_xywh]
                area = bbox_xywh[2]*bbox_xywh[3]
            else:
                bbox_xywh = None
                area = None
            #
            category_id = classes_selected.index(label_mapped) + category_id_start
            annotation_info = {
                'bbox': bbox_xywh,
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id
            }
            annotations.append(annotation_info)
            annotation_id += 1
        #
        image_id += 1
    #
#

info = {
    "description": "TIScapes2017 Driving Dataset",
    "url": None,
    "version": "1.0",
    "year": 2017,
    "contributor": "Texas instruments",
    "date_created": "2017/feb/07"
}

dataset_store = dict(info=info, categories=categories, images=images, annotations=annotations)

with open(os.path.join(dest_dir, f'instances.json'), 'w') as jfp:
    json.dump(dataset_store, jfp)
#