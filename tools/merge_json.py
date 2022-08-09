import os
import mmcv
import yaml
import shutil
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser("LINEMOD_PBR2COCO_PARSER")
parser.add_argument("--json1", default="/data/ssd/6d_pose/ycb/annotations/real/instances_train.json", type=str, help="First json file to merge")
parser.add_argument("--json2", default="/data/ssd/6d_pose/ycb/annotations/pbr/instances_train.json", type=str, help="Second json file to merge")
parser.add_argument("--outfile", default="/data/ssd/6d_pose/ycb/annotations/instances_train.json", type=str, help="Merged file")
args = parser.parse_args()

assert args.json1 != None, "Please provide the first json file"
assert args.json2 != None, "Please provide the second json file"

def merge_jsons():

    with open(args.json1) as foo:
        print("loading json1...")
        coco_json1 = json.load(foo)
        json1_img_count = coco_json1['images'][-1]['id']
        json1_obj_count = coco_json1['annotations'][-1]['id']

    with open(args.json2) as foo:
        print("loading json2...")
        coco_json2 = json.load(foo)

    print("Updating image id")
    for image_dict in coco_json2["images"]:
        image_dict["id"] = image_dict["id"] + json1_img_count

    print("Updating object id")
    for annotation_dict in coco_json2["annotations"]:
        annotation_dict["image_id"] = annotation_dict["image_id"] + json1_img_count
        annotation_dict["id"] = annotation_dict["id"] + json1_obj_count

    coco_json1["annotations"].extend(coco_json2["annotations"])
    coco_json1["images"].extend(coco_json2["images"])
    mmcv.dump(coco_json1, args.outfile)



if __name__ == "__main__":
        merge_jsons()
