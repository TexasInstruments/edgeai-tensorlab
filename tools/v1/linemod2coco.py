import os
import mmcv
import yaml
import shutil
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser("LINEMOD2COCO_PARSER")
parser.add_argument("--basepath", default="./data/lmo", type=str, help="path to ycbv dataset")
parser.add_argument("--split", default='train', type=str, help="split can be wither train or test")

#parser.add_argument("--keyframes", default="./data/ycbv/keyframe.txt", type=str, help="path to the keyframes file list")

parser.add_argument("--lm", default=False, action="store_true", help="Select only LM classes")
parser.add_argument("--lmo", default=False, action="store_true", help="Select only LMO classes")
parser.add_argument("--lmob", default=False, action="store_true", help="Select only LMO classes including benchvise")

args = parser.parse_args()

lme = [3, 7]  #linemod excluded , 15-2=13
lmoe = [2, 3, 4, 7, 13, 14, 15]  #15-7=8
lmoeb = [3, 4, 7, 13, 14, 15]  #15-6=9


def convert_to_coco_json(split='train', type='real'):
    if split == 'train':
        basepath = os.path.join(args.basepath, '{}_{}'.format(split, type))
        outfile = os.path.join(args.basepath, 'annotations', 'instances_{}_{}.json'.format(split, type))
    else:
        basepath = os.path.join(args.basepath, '{}_{}'.format(split, keyframes))
        outfile = os.path.join(args.basepath, 'annotations', 'instances_{}_{}.json'.format(split, keyframes))
    data_folders = sorted(os.listdir(basepath))

    image_count = 0
    obj_count = 0

    for data_folder_idx, data_folder in enumerate(data_folders):
        if args.lm:
            if int(data_folder) in lme: continue
        if args.lmo:
            if int(data_folder) in lmoe: continue
        if args.lmob:
            if int(data_folder) in lmoeb: continue

        data_path = os.path.join(basepath, data_folder)
        print("Loading {}".format(data_path + '/gt.yml'))
        with open(data_path + '/gt.yml') as yaml_file:
            gt_dict = yaml.safe_load(yaml_file)
        print("Loading completed {}".format(data_path + '/gt.yml'))

        with open(data_path + '/test.txt', 'r') as f:
            test_list = list(f)
            test_list = [idx.rstrip() for idx in test_list]

        if not merge or data_folder_idx==0:
            coco_train = dict()
            coco_train["images"] = []
            coco_train["type"] = "instance"
            coco_train["categories"] = []
            coco_train["annotations"] = []

            coco_test = dict()
            coco_test["images"] = []
            coco_test["type"] = "instance"
            coco_test["categories"] = []
            coco_test["annotations"] = []

            class_to_name = {0: "ape", 1: "benchvise", 2: "bowl", 3: "can", 4: "cat", 5: "cup", 6: "driller", 7: "duck", 8: "glue", 9: "holepuncher", 10: "iron", 11: "lamp", 12: "phone", 13: "cam", 14: "eggbox"}
            for obj_class in class_to_name:
                category = dict([
                    ("supercategory", "object"),
                    ("id", obj_class),
                    ("name", class_to_name[obj_class])
                ])

                coco_train["categories"].append(category)
                coco_test["categories"].append(category)

        pbar = tqdm(enumerate(gt_dict.items()), total=len(gt_dict))
        for image_num, objects in pbar:
            filename = "{:04}".format(image_num) + '.png'
            height, width = mmcv.imread(data_path + '/rgb/' + filename).shape[:2]
            image = dict([
                ("image_folder", data_folder),
                ("id", image_count),
                ("file_name", filename),
                ("height", height),
                ("width", width),
                ("type", "lm_real" )
            ])

            coco_test["images"].append(image) if filename[:-4] in test_list else coco_train["images"].append(image)

            for object in objects[1]:
                annotation = dict([
                    ("image_id", image_count),
                    ("id", obj_count),
                    ("bbox", object["obj_bb"]),
                    ("area", object["obj_bb"][2] * object["obj_bb"][3]),
                    ("iscrowd", 0),
                    ("category_id", object["obj_id"] - 1),
                    ("R", object["cam_R_m2c"]),
                    ("T", object["cam_t_m2c"])
                ])
                obj_count += 1
                coco_test["annotations"].append(annotation) if filename[:-4] in test_list else coco_train["annotations"].append(annotation)
            image_count += 1
        pbar.close()
        if not merge:
            outfile_train = '/data/ssd/6d_pose/LINEMOD_COCO/instances_train_{}.json'.format(data_folder)
            outfile_test = '/data/ssd/6d_pose/LINEMOD_COCO/instances_test_{}.json'.format(data_folder)
            mmcv.dump(coco_train, outfile_train)
            mmcv.dump(coco_test, outfile_test)

    if merge:
        mmcv.dump(coco_train, outfile_train_merged)
        mmcv.dump(coco_test, outfile_test_merged)


if __name__ == "__main__":
        convert_to_coco_json(merge=True)
