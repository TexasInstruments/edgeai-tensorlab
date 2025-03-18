import os
import mmcv
from tqdm import tqdm
import json
import argparse
from merge_json import merge_jsons
import copy

parser = argparse.ArgumentParser("TLESS2COCO_PARSER")
parser.add_argument("--datapath", default="./datasets/tless", type=str, help="path to ycbv dataset")
parser.add_argument("--keyframes", default="./data/tless/keyframe.txt", type=str, help="path to the keyframes file list")
parser.add_argument("--split", default='train', type=str, help="split can be either train or test")

args = parser.parse_args()
num_classes = 30
class_to_name = {}
#T-Less data has  class doesn't have any name. Hence, providing generic names: class_00, class_01,..., class_29
for i in range(num_classes):
    class_to_name [i] =  "class_{:02}".format(i)

def create_camera_json(datapath="./datasets/tless"):
    """
    Create camera json for diffrent sequences and dump it into datapath.
    This ensures all camera parameters are at the same location.
    """
    #camera_primesense
    camera_primesense_path = os.path.join(datapath, 'camera_primesense.json')
    with open(camera_primesense_path) as foo:
        camear_primesense = json.load(foo)

    #camera_pbr
    camera_pbr_path = os.path.join(datapath, 'train_pbr/000001/scene_camera.json')
    camera_pbr_out_path = os.path.join(datapath, 'camera_train_pbr.json')
    if not os.path.exists(camera_pbr_out_path):
        with open(camera_pbr_path) as foo:
            camear_pbr = json.load(foo)['0']['cam_K']
            camera_pbr_out = copy.deepcopy(camear_primesense)
            camera_pbr_out['cx'] = camear_pbr[2]
            camera_pbr_out['cy'] = camear_pbr[5]
            camera_pbr_out['width'] = int(2 * camear_pbr[2])
            camera_pbr_out['height'] = int(2 * camear_pbr[5])
            print("Generating camera_train_pbr.json")
            mmcv.dump(camera_pbr_out, camera_pbr_out_path)

    # camera_train_real
    camera_train_real_path = os.path.join(datapath, 'train_real/000001/scene_camera.json')
    camera_train_real_out_path = os.path.join(datapath, 'camera_train_real.json')
    if not os.path.exists(camera_train_real_out_path):
        with open(camera_train_real_path) as foo:
            camear_train_real = json.load(foo)['0']['cam_K']
            camera_train_real_out = copy.deepcopy(camear_primesense)
            camera_train_real_out['cx'] = camear_train_real[2]
            camera_train_real_out['cy'] = camear_train_real[5]
            camera_train_real_out['width'] = 400
            camera_train_real_out['height'] = 400
            print("Generating camera_train_real.json")
            mmcv.dump(camera_train_real_out, camera_train_real_out_path)

    #camera_test_bop
    camera_test_bop_path = os.path.join(datapath, 'test_bop/000001/scene_camera.json')
    camera_test_bop_out_path = os.path.join(datapath, 'camera_test_bop.json')
    if not os.path.exists(camera_test_bop_out_path):
        with open(camera_test_bop_path) as foo:
            camera_test_bop = json.load(foo)['1']['cam_K']
            camera_test_bop_out = copy.deepcopy(camear_primesense)
            camera_test_bop_out['cx'] = camera_test_bop[2]
            camera_test_bop_out['cy'] = camera_test_bop[5]
            camera_test_bop_out['width'] = 720
            camera_test_bop_out['height'] = 540
            print("Generating camera_test_bop.json")
            mmcv.dump(camera_test_bop_out, camera_test_bop_out_path)


def convert_tless2coco(split='train', type='real', keyframes=None, datapath="./datasets/tless"):
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
                    filename = "{:06}".format(image_index) + '.png'
            elif type == "pbr":
                filename = "{:06}".format(image_index) + '.jpg'

            if keyframes !='bop':
                if split == 'test':
                    if os.path.join(data_folder, filename) not in keyframes_list :
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
    create_camera_json(args.datapath)
    if args.split == "train":
        #Generate annotations in COCO format for real training images
        print("Train: Generating annotation for real images")
        json_real = convert_tless2coco(split=args.split, type='real', datapath=args.datapath)
        # Generate annotations in COCO format for PBR training images
        print("Train: Generating annotation for PBR images")
        json_pbr = convert_tless2coco(split=args.split, type='pbr', keyframes='bop', datapath=args.datapath)
        #Merge real and pbr annotation for the final training annotation
        train_annotations = os.path.join(args.datapath, 'annotations', 'instances_{}.json'.format(args.split))
        merge_jsons(json_real, json_pbr, train_annotations)
    elif args.split=="test":
        convert_tless2coco(split=args.split, type='real', keyframes='bop', datapath=args.datapath)
        # convert_to_coco_json(split=args.split, type='real', keyframes='all')
    else:
        print("Invalid split given, Only vaiid options are \'train\' and \'test\'")

