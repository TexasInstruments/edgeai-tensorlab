# Copyright (c) 2018-2022, Texas Instruments Incorporated
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Dataset Location: http://host.robots.ox.ac.uk/pascal/VOC/
# The PASCAL Visual Object Classes (VOC) Challenge
# Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
# International Journal of Computer Vision, 88(2), 303-338, 2010


import json
import os
import datetime
import shutil
import scipy
import PIL

from .... import utils
from . import dataset_utils


def get_categories(state, example_project_path):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')
    coco_dict = json.load(example_project_annotation_file_name)
    return coco_dict


def dataset_paths(state, example_project_path, **kwargs):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')
    return (example_project_files_path,example_project_annotation_file_name)


def dataset_reload(state, example_project_path, force_download=False, log_writer=None,
                   progressbar_creator=None):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    shutil.rmtree(example_dataset_path, ignore_errors=True)
    return dataset_download(state, example_project_path, force_download, log_writer=log_writer,
                     progressbar_creator=progressbar_creator)


def dataset_download(state, example_project_path, force_download=False, log_writer=None,
                     progressbar_creator=None):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')

    print(f'''
        Downloading and preparing dataset, this may take some time.
        *   Nilsback, M-E. and Zisserman, A.
        *   Automated flower classification over a large number of classes  
        *   Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008)
        *   https://www.robots.ox.ac.uk/~vgg/data/flowers/102/        
        ''')

    extract_root = os.path.join(example_project_path, 'other', 'extract')
    coco_train_images_path = os.path.join(extract_root, 'train2017')
    coco_val_images_path = os.path.join(extract_root, 'val2017')
    if (not force_download) and os.path.exists(extract_root) and \
        os.path.exists(coco_train_images_path) and os.path.exists(coco_val_images_path) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'Dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
                    'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
                    'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat']

    download_root = os.path.join(example_project_path, 'other', 'download')
    download_success, message, download_paths = utils.download_files(dataset_urls, download_root=download_root,
                                                    extract_root=extract_root, log_writer=log_writer,
                                                    progressbar_creator=progressbar_creator)

    # prepare the images
    os.makedirs(example_project_files_path, exist_ok=True)
    os.makedirs(example_project_annotations_path, exist_ok=True)

    labels_list = scipy.io.loadmat(download_paths[1])['labels'][0]
    split_list = scipy.io.loadmat(download_paths[2])
    split_names_dataset = ['trnid', 'valid', 'tstid']
    split_names_json = ['train', 'val', 'test']

    dataset_info = {"year": 2008, "version": "1.0", "description": "", "contributor": "Nilsback, M-E. and Zisserman, A.",
                    "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/", "date_created": ""}
    dataset_categories = []
    unique_categories = [cat for cat in labels_list]
    unique_categories = list(set(unique_categories))
    # id here starts from 1.
    # TODO: should we start from 0?
    for cat_id, category in enumerate(unique_categories):
        assert category == int(cat_id+1), 'invalid category'
        cat = dict(id=int(cat_id+1), name=str(category))
        dataset_categories.append(cat)
    #

    image_dir = os.path.join(extract_root, 'jpg')
    dataset_images = []
    dataset_annotations = []
    annotation_id = 0
    for split_id, split_name_dataset in enumerate(split_names_dataset[:2]):
        split_name_json = split_names_json[split_id]
        num_images = 0
        for file_id, image_id in enumerate(split_list[split_name_dataset][0]):
            image_id = int(image_id)
            image_file = "image_%05d.jpg" % image_id
            src_image = os.path.join(image_dir, image_file)
            dst_image = os.path.join(example_project_files_path, image_file)
            shutil.copy2(src_image, dst_image)

            # image info
            pil_image = PIL.Image.open(dst_image)
            image_size = pil_image.size
            image_info = {"width": image_size[0], "height": image_size[1], "id": image_id, "file_name": image_file,
                          'split_name': split_name_json}
            dataset_images.append(image_info)

            # annotation info
            category_id = int(labels_list[image_id-1])
            assert category_id in unique_categories, 'invalid category found'
            annotation_info = {"id": annotation_id, "image_id": image_id, "category_id": category_id, "segmentation": [],
                                "bbox": None, "ignore": 0, "iscrowd": 0, "area": 0}
            dataset_annotations.append(annotation_info)
            annotation_id += 1
            num_images += 1
        #
        print(f'{split_name_dataset} : {num_images}')
    #
    dataset_dict = dict(info=dataset_info,
                         categories=dataset_categories,
                         images=dataset_images,
                         annotations=dataset_annotations)
    with open(example_project_annotation_file_name, 'w') as afp:
        json.dump(dataset_dict, afp)
    #
    return (example_project_files_path,example_project_annotation_file_name)
