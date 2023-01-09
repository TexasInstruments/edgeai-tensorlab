import os
import mmcv
import yaml
import shutil
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser("LINEMOD2COCO_PARSER")
parser.add_argument("--datapath", default="./datasets/lmo", type=str, help="path to ycbv dataset")
parser.add_argument("--split", default='train', type=str, help="split can be wither train or test")


args = parser.parse_args()

class_to_name = {0: "ape", 1: "benchvise", 2: "bowl", 3: "can", 4: "cat",
                 5: "cup", 6: "driller", 7: "duck", 8: "glue", 9: "holepuncher",
                 10: "iron", 11: "lamp", 12: "phone", 13: "cam", 14: "eggbox"}

def convert_lmo2coco(split='train', type='real', keyframes=None, datapath="./datasets/lmo"):
    if split == 'train':
        basepath = os.path.join(datapath, '{}_{}'.format(split, type))
        outfile = os.path.join(datapath, 'annotations', 'instances_{}_{}.json'.format(split, type))
    else:
        basepath = os.path.join(datapath, '{}_{}'.format(split, keyframes))
        outfile = os.path.join(datapath, 'annotations', 'instances_{}_{}.json'.format(split, keyframes))
    data_folders = sorted(os.listdir(basepath))

    for data_folder_idx, data_folder in enumerate(data_folders):
        data_path = os.path.join(basepath, data_folder)
        annotations_gt = dict()
        for f in ('scene_gt_info.json', 'scene_gt.json'):
            path = os.path.join(data_path,  f)
            if os.path.exists(path):
                print("Loading {}".format(path))
                with open(path) as foo:
                    annotations_gt[f.split('.')[0]] = json.load(foo)

        if data_folder_idx==0:
            coco = dict()
            coco["images"] = []
            coco["type"] = "instance"
            coco["categories"] = []
            coco["annotations"] = []

            for obj_class in class_to_name:
                category = dict([
                    ("supercategory", "object"),
                    ("id", obj_class),
                    ("name", class_to_name[obj_class])
                ])

                coco["categories"].append(category)

            obj_count = 0
            img_count = 0

        pbar = tqdm(enumerate(zip(list(annotations_gt['scene_gt'].items()), list(annotations_gt['scene_gt_info'].items()))), total=len(annotations_gt['scene_gt_info']))
        for image_index, objects in pbar:
            objects_gt, objects_gt_info = objects[0], objects[1]
            if type == "real":
                if keyframes == 'bop':
                    filename = "{:06}".format(int(objects_gt[0])) + '.png'
                else:
                    filename = "{:06}".format(image_index+1) + '.png'
            elif type == "pbr":
                filename = "{:06}".format(image_index) + '.jpg'

            height, width = mmcv.imread(data_path + '/rgb/' + filename).shape[:2]
            image = dict([
                ("image_folder", data_folder),
                ("id", img_count), #
                ("file_name", filename),
                ("height", height),
                ("width", width),
            ])

            coco["images"].append(image)
            for object_gt, object_gt_info  in zip(objects_gt[1], objects_gt_info[1]):
                if object_gt_info['visib_fract'] > 0:
                    annotation = dict([
                        ("image_id", img_count),
                        ("id", obj_count),
                        ("bbox", object_gt_info["bbox_visib"]),
                        ("area", object_gt_info["bbox_visib"][2] * object_gt_info["bbox_visib"][3]),
                        ("iscrowd", 0),
                        ("category_id", object_gt["obj_id"]-1),
                        ("R", object_gt["cam_R_m2c"]),
                        ("T", object_gt["cam_t_m2c"])
                    ])
                    obj_count += 1
                    coco["annotations"].append(annotation)
            img_count += 1
        pbar.close()
    mmcv.dump(coco, outfile)
    return outfile


if __name__ == "__main__":
    convert_lmo2coco(split='train', type='pbr',keyframes=None, datapath=args.datapath)
    convert_lmo2coco(split='test', type='real',keyframes='bop', datapath=args.datapath)

