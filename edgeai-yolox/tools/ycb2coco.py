import os
import mmcv
from tqdm import tqdm
import json
import argparse
from merge_json import merge_jsons

parser = argparse.ArgumentParser("YCBV2COCO_PARSER")
parser.add_argument("--datapath", default="./datasets/ycbv", type=str, help="path to ycbv dataset")
parser.add_argument("--keyframes", default="./data/ycbv/keyframe.txt", type=str, help="path to the keyframes file list")
parser.add_argument("--split", default='train', type=str, help="split can be either train or test")

args = parser.parse_args()

class_to_name = {
    0: "002_master_chef_can" , 1: "003_cracker_box" ,  2: "004_sugar_box" , 3: "005_tomato_soup_can",  4: "006_mustard_bottle",
    5: "007_tuna_fish_can",  6: "008_pudding_box" , 7: "009_gelatin_box", 8: "010_potted_meat_can",  9: "011_banana",
    10: "019_pitcher_base", 11: "021_bleach_cleanser",  12: "024_bowl", 13: "025_mug", 14: "035_power_drill",
    15: "036_wood_block", 16: "037_scissors", 17: "040_large_marker", 18: "051_large_clamp", 19: "052_extra_large_clamp",
    20: "061_foam_brick"
}


def convert_ycb2coco(split='train', type='real', keyframes=None, datapath="./datasets/ycbv"):
    if split == 'train':
        basepath = os.path.join(datapath, '{}_{}'.format(split, type))
        outfile = os.path.join(datapath, 'annotations', 'instances_{}_{}.json'.format(split, type))
    else:
        basepath = os.path.join(datapath, '{}_{}'.format(split, keyframes))
        outfile = os.path.join(datapath, 'annotations', 'instances_{}_{}.json'.format(split, keyframes))
    data_folders = sorted(os.listdir(basepath))


    if split == 'test':
        if keyframes != 'bop':
            keyframes = os.path.join(datapath, 'keyframe.txt')
            assert os.path.exists(keyframes), "keyframe file :{} is not present".format(keyframes)
            with open(keyframes):
                keyframes_list = list(open(keyframes))
            keyframes_list = ["00" + keyframe.rstrip()+".png" for keyframe in keyframes_list]

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

            if keyframes !='bop':
                if split == 'test':
                    if os.path.join(data_folder, filename) not in keyframes_list :
                        continue
                elif image_index%10 != 0:
                    continue

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
    if args.split == "train":
        #Generate annotations in COCO format for real training images
        print("Train: Generating annotation for real images")
        json_real = convert_ycb2coco(split=args.split, type='real', datapath=args.datapath)
        # Generate annotations in COCO format for PBR training images
        print("Train: Generating annotation for PBR images")
        json_pbr = convert_ycb2coco(split=args.split, type='pbr', keyframes='bop', datapath=args.datapath)
        #Merge real and pbr annotation for the final training annotation
        train_annotations = os.path.join(args.datapath, 'annotations', 'instances_{}.json'.format(args.split))
        merge_jsons(json_real, json_pbr, train_annotations)
    elif args.split=="test":
        convert_ycb2coco(split=args.split, type='real', keyframes='bop', datapath=args.datapath)
        # convert_to_coco_json(split=args.split, type='real', keyframes='all')
    else:
        print("Invalid split given, Only vaiid options are \'train\' and \'test\'")

