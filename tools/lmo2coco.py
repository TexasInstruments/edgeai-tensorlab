import os
import mmcv
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser("LINEMOD2COCO_PARSER")
parser.add_argument("--datapath", default="./datasets/lmo", type=str, help="path to lmo dataset")
parser.add_argument("--split", default='train', type=str, help="split can be wither train or test")


args = parser.parse_args()

class_to_name_orig = {0: "ape", 4: "can", 5: "cat", 7: "driller",
                      8: "duck", 9: "eggbox", 10: "glue", 11: "holepuncher"}

class_to_name = { 0: "ape", 1: "can", 2: "cat", 3: "driller",
                  4: "duck", 5: "eggbox", 6: "glue", 7: "holepuncher"}

class_map = {                            #class map used for training
             0: 0, 4: 1, 5: 2, 7: 3,
             8: 4, 9: 5, 10: 6, 11: 7,
             }

def convert_lmo2coco(split='train', type='real', keyframes=None, datapath="./datasets/lmo"):
    if split == 'train':
        basepath = os.path.join(datapath, '{}_{}'.format(split, type))
        outfile = os.path.join(datapath, 'annotations', 'instances_{}.json'.format(split))
        # outfile = os.path.join(datapath, 'annotations', 'instances_{}_{}.json'.format(split, type))
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
            if "real" in path:
                image.update({'type': "real"})
            elif "pbr" in path:
                image.update({'type': "pbr"})
            elif "bop" in path :
                image.update({'type': "bop"})
            elif "all" in path:
                image.update({'type': "all"})
            else:
                image.update({'type': "syn"})

            coco["images"].append(image)
            for object_gt, object_gt_info  in zip(objects_gt[1], objects_gt_info[1]):
                if object_gt["obj_id"]-1 not in class_to_name_orig.keys() : continue  # Consider classes that are part of lmo
                if object_gt_info['visib_fract'] > 0:
                    annotation = dict([
                        ("image_id", img_count),
                        ("id", obj_count),
                        ("bbox", object_gt_info["bbox_visib"]),
                        ("area", object_gt_info["bbox_visib"][2] * object_gt_info["bbox_visib"][3]),
                        ("iscrowd", 0),
                        ("category_id", class_map[object_gt["obj_id"]-1]),
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
    if args.split=='train':
        #For LMO, we only use pbr images for training since no real imaeges as part of training have all annotation available.
        convert_lmo2coco(split='train', type='pbr',keyframes=None, datapath=args.datapath)
    elif args.split=='test':
        convert_lmo2coco(split='test', type='real',keyframes='bop', datapath=args.datapath)
    else:
        print("Invalid split given, Only vaiid options are \'train\' and \'test\'")

