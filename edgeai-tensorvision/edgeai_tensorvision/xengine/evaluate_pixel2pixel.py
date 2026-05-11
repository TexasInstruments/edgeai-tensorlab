# Copyright (c) 2018-2021, Texas Instruments
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

import sys
import torch

import numpy as np
import cv2

import edgeai_torchmodelopt
from edgeai_torchmodelopt import xnn


##################################################
# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

##################################################
np.set_printoptions(precision=3)

##################################################
def get_config():
    args = xnn.utils.ConfigNode()

    args.data_path = './'                   # path to dataset
    args.num_classes = None                 # number of classes (for segmentation)

    args.save_path = None        # checkpoints save path

    args.img_resize = None                  # image size to be resized to

    args.output_size = None                 # target output size to be resized to

    args.upsample_mode = 'bilinear'         # 'upsample mode to use. choices=['nearest','bilinear']

    args.eval_semantic = False              # 'set for 19 class segmentation

    args.eval_semantic_five_class = False   # set for five class segmentation

    args.eval_motion = False                # set for motion segmentation

    args.eval_depth = False                 # set for motion segmentation

    args.scale_factor = 1.0                 # scale_factor used by Deepak to improve depth accuracy
    args.verbose = True                     # whether to print scores for all frames or the final result

    args.inf_suffix = ''                    # suffix for diffrent job sem/mot/depth
    args.frame_IOU = False                  # Check for framewise IOU for segmentation
    args.phase = 'validation'
    args.date = None
    return args


# ################################################
def main(args):

    assert not hasattr(args, 'model_surgery'), 'the argument model_surgery is deprecated, it is not needed now - remove it'

    ################################
    # print everything for log
    print('=> args: ', args)

    #################################################
    if args.eval_semantic:
        args.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        args.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        args.class_map = dict(zip(args.valid_classes, range(19)))
        args.gt_suffix = '_labelIds.png'
        if args.inf_suffix == '':
          args.inf_suffix = '.png'
        args.output_channels = 19
    elif args.eval_semantic_five_class:
        args.void_classes = [-1, 255]
        args.valid_classes = [0, 1, 2, 3, 4]
        args.class_map = dict(zip(args.valid_classes, range(5)))
        args.gt_suffix = '_labelTrainIds.png'
        if args.inf_suffix == '':        
          args.inf_suffix = '1.png'
        args.output_channels = 5
    elif args.eval_motion:
        args. void_classes = []
        args.valid_classes = [0, 255]
        args.class_map = dict(zip(args.valid_classes, range(2)))
        args.gt_suffix = '_labelTrainIds_motion.png'
        if args.inf_suffix == '':        
          args.inf_suffix = '2.png'
        args.output_channels = 2
    elif args.eval_depth:
        args. void_classes = []
        args.valid_classes = [0, 255]
        args.gt_suffix = '.png'
        if args.inf_suffix == '':        
          args.inf_suffix = '0.png'
        args.max_depth = 20

    print(args)
    print("=> fetching gt labels in '{}'".format(args.label_path))
    label_files = xnn.utils.recursive_glob(rootdir=args.label_path, suffix=args.gt_suffix)
    label_files.sort()
    print('=> {} gt label samples found'.format(len(label_files)))

    print("=> fetching inferred images in '{}'".format(args.infer_path))
    infer_files = xnn.utils.recursive_glob(rootdir=args.infer_path, suffix=args.inf_suffix)
    infer_files.sort()
    print('=> {} inferred samples found'.format(len(infer_files)))

    assert len(label_files) == len(infer_files), 'Number of label files and inference file must be same'

    #################################################
    if not args.eval_depth:
        validate(args, label_files, infer_files)
    else:
        validate_depth(args, label_files, infer_files)


def validate(args, label_files, infer_files):

    confusion_matrix = np.zeros((args.output_channels, args.output_channels+1))

    for iter, (label_file, infer_file) in enumerate(zip(label_files, infer_files)):
        if args.frame_IOU:
            confusion_matrix = np.zeros((args.output_channels, args.output_channels + 1))
        gt = encode_segmap(args, cv2.imread(label_file, 0))
        inference = cv2.imread(infer_file)
        inference = inference[:,:,-1]
        if inference.shape != gt.shape:
            inference = np.expand_dims(np.expand_dims(inference, 0),0)
            inference = torch.tensor(inference).float()
            scale_factor = torch.tensor(args.scale_factor).float()
            inference = inference/scale_factor
            inference = torch.nn.functional.interpolate(inference, (gt.shape[0], gt.shape[1]), mode='nearest')
            inference = np.ndarray.astype(np.squeeze(inference.numpy()), dtype=np.uint8)
        confusion_matrix = eval_output(args, inference, gt, confusion_matrix, args.output_channels)
        accuracy, mean_iou, iou, f1_score= compute_accuracy(args, confusion_matrix, args.output_channels)
        if args.verbose:
            print('{}/{}inferred image {} mIOU {}'.format(iter, len(label_files), label_file.split('/')[-1], mean_iou))
            if iter % 100 ==0:
                print('\npixel_accuracy={},\nmean_iou={},\niou={},\nf1_score = {}'.format(accuracy, mean_iou, iou, f1_score))
                sys.stdout.flush()
    print('\npixel_accuracy={},\nmean_iou={},\niou={},\nf1_score = {}'.format(accuracy, mean_iou, iou, f1_score))
    #

def encode_segmap(args, mask):
    ignore_index = 255
    for _voidc in args.void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in args.valid_classes:
        mask[mask == _validc] = args.class_map[_validc]
    return mask


def eval_output(args, output, label, confusion_matrix, n_classes):
    if len(label.shape)>2:
        label = label[:,:,0]
    gt_labels = label.ravel()
    det_labels = output.ravel().clip(0,n_classes)
    gt_labels_valid_ind = np.where(gt_labels != 255)
    gt_labels_valid = gt_labels[gt_labels_valid_ind]
    det_labels_valid = det_labels[gt_labels_valid_ind]
    for r in range(confusion_matrix.shape[0]):
        for c in range(confusion_matrix.shape[1]):
            confusion_matrix[r,c] += np.sum((gt_labels_valid==r) & (det_labels_valid==c))

    return confusion_matrix


def compute_accuracy(args, confusion_matrix, n_classes):
    num_selected_classes = n_classes
    tp = np.zeros(n_classes)
    population = np.zeros(n_classes)
    det = np.zeros(n_classes)
    iou = np.zeros(n_classes)
    
    for r in range(n_classes):
      for c in range(n_classes):
        population[r] += confusion_matrix[r][c]
        det[c] += confusion_matrix[r][c]   
        if r == c:
          tp[r] += confusion_matrix[r][c]

    for cls in range(num_selected_classes):
      intersection = tp[cls]
      union = population[cls] + det[cls] - tp[cls]
      iou[cls] = (intersection / union) if union else 0     # For caffe jacinto script
      #iou[cls] = (intersection / (union + np.finfo(np.float32).eps))  # For pytorch-jacinto script

    num_nonempty_classes = 0
    for pop in population:
      if pop>0:
        num_nonempty_classes += 1
          
    mean_iou = np.sum(iou) / num_nonempty_classes if num_nonempty_classes else 0
    accuracy = np.sum(tp) / np.sum(population) if np.sum(population) else 0
    
    #F1 score calculation
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)

    for cls in range(num_selected_classes):
        fp[cls] = det[cls] - tp[cls]
        fn[cls] = population[cls] - tp[cls]
        precision[cls] = tp[cls] / (det[cls] + 1e-10)
        recall[cls] = tp[cls] / (tp[cls] + fn[cls] + 1e-10)        
        f1_score[cls] = 2 * precision[cls]*recall[cls] / (precision[cls] + recall[cls] + 1e-10)

    return accuracy, mean_iou, iou, f1_score

def validate_depth(args, label_files, infer_files):
    max_depth = args.max_depth
    print("Max depth set to {} meters".format(max_depth))
    ard_err = None
    for iter, (label_file, infer_file) in enumerate(zip(label_files, infer_files)):
        gt = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        inference = cv2.imread(infer_file, cv2.IMREAD_UNCHANGED)

        if inference.shape != gt.shape:
            inference = np.expand_dims(np.expand_dims(inference, 0),0)
            inference = torch.tensor(inference).float()
            scale_factor = torch.tensor(args.scale_factor).float()
            inference = (inference/scale_factor).int().float()
            inference = torch.nn.functional.interpolate(inference, (gt.shape[0], gt.shape[1]), mode='nearest')
            inference = torch.squeeze(inference)

        gt[gt==255] = 0
        gt[gt>max_depth]=max_depth
        gt = torch.tensor(gt).float()
        inference[inference > max_depth] = max_depth

        valid = (gt!=0)
        gt = gt[valid]
        inference = inference[valid]
        if len(gt)>2:
            if ard_err is None:
                ard_err = [absreldiff_rng3to80(inference, gt).mean()]
            else:
                ard_err.append(absreldiff_rng3to80(inference, gt).mean())
        elif len(gt) < 2:
            if ard_err is None:
                ard_err = [0.0]
            else:
                ard_err.append(0.0)

        if args.verbose:
            print('{}/{} Inferred Frame {} ARD {}'.format(iter+1, len(label_files), label_file.split('/')[-1], float(ard_err[-1])))

    print('ARD_final {}'.format(float(torch.tensor(ard_err).mean())))

def absreldiff(x, y, eps = 0.0, max_val=None):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    if max_val is not None:
        x = torch.clamp(x, -max_val, max_val)
        y = torch.clamp(y, -max_val, max_val)
    #

    diff = torch.abs(x - y)
    y = torch.abs(y)

    den_valid = (y == 0).float()
    eps_arr = (den_valid * (1e-6))   # Just to avoid divide by zero

    large_arr = (y > eps).float()    # ARD is not a good measure for small ref values. Avoid them.
    out = (diff / (y + eps_arr)) * large_arr
    return out

def absreldiff_rng3to80(x, y):
    return absreldiff(x, y, eps = 3.0, max_val=80.0)


if __name__ == '__main__':
    train_args = get_config()
    main(train_args)
