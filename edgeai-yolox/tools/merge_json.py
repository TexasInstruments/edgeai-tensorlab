import os
import mmcv
import json

def merge_jsons(json1, json2, outfile):
    assert os.path.exists(json1), "{} doesn't exist".format(json1)
    assert os.path.exists(json2), "{} doesn't exist".format(json2)

    with open(json1) as foo:
        print("loading {}...".format(json1))
        coco_json1 = json.load(foo)
        json1_img_count = coco_json1['images'][-1]['id']
        json1_obj_count = coco_json1['annotations'][-1]['id']

    with open(json2) as foo:
        print("loading {}...".format(json2))
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
    mmcv.dump(coco_json1, outfile)


# if __name__ == "__main__":
#         merge_jsons(args.json1 , args.json2, args.outfile)
