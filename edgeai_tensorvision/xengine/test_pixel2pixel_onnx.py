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

import os
import time
import sys
import warnings

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import datetime
import numpy as np
import random
import cv2
import PIL
import PIL.Image

import onnx
import caffe2
import caffe2.python.onnx.backend

from edgeai_torchmodelopt import xnn
from edgeai_tensorvision import xvision



# ################################################
def get_config():
    args = xnn.utils.ConfigNode()
    args.model_config = xnn.utils.ConfigNode()
    args.dataset_config = xnn.utils.ConfigNode()

    args.dataset_name = 'flying_chairs'              # dataset type
    args.model_name = 'flownets'                # model architecture, overwritten if pretrained is specified: '

    args.data_path = './data/datasets'                       # path to dataset
    args.save_path = None            # checkpoints save path
    args.pretrained = None

    args.model_config.output_type = ['flow']                # the network is used to predict flow or depth or sceneflow')
    args.model_config.output_channels = None                 # number of output channels
    args.model_config.input_channels = None                  # number of input channels
    args.n_classes = None                       # number of classes (for segmentation)

    args.logger = None                          # logger stream to output into

    args.prediction_type = 'flow'               # the network is used to predict flow or depth or sceneflow
    args.split_file = None                      # train_val split file
    args.split_files = None                     # split list files. eg: train.txt val.txt
    args.split_value = 0.8                      # test_val split proportion (between 0 (only test) and 1 (only train))

    args.workers = 8                            # number of data loading workers

    args.epoch_size = 0                         # manual epoch size (will match dataset size if not specified)
    args.epoch_size_val = 0                     # manual epoch size (will match dataset size if not specified)
    args.batch_size = 8                         # mini_batch_size
    args.total_batch_size = None                # accumulated batch size. total_batch_size = batch_size*iter_size
    args.iter_size = 1                          # iteration size. total_batch_size = batch_size*iter_size

    args.tensorboard_num_imgs = 5               # number of imgs to display in tensorboard
    args.phase = 'validation'                   # evaluate model on validation set
    args.pretrained = None                      # path to pre_trained model
    args.date = None                            # don\'t append date timestamp to folder
    args.print_freq = 10                        # print frequency (default: 100)

    args.div_flow = 1.0                         # value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results
    args.milestones = [100,150,200]             # epochs at which learning rate is divided by 2
    args.losses = ['supervised_loss']           # loss functions to minimize
    args.metrics = ['supervised_error']         # metric/measurement/error functions for train/validation
    args.class_weights = None                   # class weights

    args.multistep_gamma = 0.5                  # steps for step scheduler
    args.polystep_power = 1.0                   # power for polynomial scheduler

    args.rand_seed = 1                          # random seed
    args.img_border_crop = None                 # image border crop rectangle. can be relative or absolute
    args.target_mask = None                      # mask rectangle. can be relative or absolute. last value is the mask value
    args.img_resize = None                      # image size to be resized to
    args.rand_scale = (1,1.25)                  # random scale range for training
    args.rand_crop = None                       # image size to be cropped to')
    args.output_size = None                     # target output size to be resized to')

    args.count_flops = True                     # count flops and report

    args.shuffle = True                         # shuffle or not
    args.is_flow = None                         # whether entries in images and targets lists are optical flow or not

    args.multi_decoder = True                   # whether to use multiple decoders or unified decoder

    args.create_video = False                   # whether to create video out of the inferred images

    args.input_tensor_name = ['0']              # list of input tensore names

    args.upsample_mode = 'nearest'              # upsample mode to use., choices=['nearest','bilinear']

    args.image_prenorm = True                   # whether normalization is done before all other the transforms
    args.image_mean = [128.0]                   # image mean for input image normalization
    args.image_scale = [1.0/(0.25*256)]         # image scaling/mult for input iamge normalization
    return args


# ################################################
# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)


# ################################################
def main(args):

    assert not hasattr(args, 'model_surgery'), 'the argument model_surgery is deprecated, it is not needed now - remove it'

    #################################################
    # global settings. rand seeds for repeatability
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    ################################
    # print everything for log
    print('=> args: ', args)

    ################################
    # args check and config
    if args.iter_size != 1 and args.total_batch_size is not None:
        warnings.warn("only one of --iter_size or --total_batch_size must be set")
    #
    if args.total_batch_size is not None:
        args.iter_size = args.total_batch_size//args.batch_size
    else:
        args.total_batch_size = args.batch_size*args.iter_size
    #

    assert args.pretrained is not None, 'pretrained onnx model path should be provided'

    #################################################
    # set some global flags and initializations
    # keep it in args for now - although they don't belong here strictly
    # using pin_memory is seen to cause issues, especially when when lot of memory is used.
    args.use_pinned_memory = False
    args.n_iter = 0
    args.best_metric = -1

    #################################################
    if args.save_path is None:
        save_path = get_save_path(args)
    else:
        save_path = args.save_path
    #

    print('=> will save everything to {}'.format(save_path))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #################################################
    if args.logger is None:
        log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        args.logger = xnn.utils.TeeLogger(filename=os.path.join(save_path,log_file))
    transforms = get_transforms(args)

    print("=> fetching img pairs in '{}'".format(args.data_path))
    split_arg = args.split_file if args.split_file else (args.split_files if args.split_files else args.split_value)
    val_dataset = xvision.datasets.__dict__[args.dataset_name](args.dataset_config, args.data_path, split=split_arg, transforms=transforms)

    print('=> {} val samples found'.format(len(val_dataset)))
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.use_pinned_memory, shuffle=args.shuffle)
    #

    #################################################
    args.model_config.output_channels = val_dataset.num_classes() if (args.model_config.output_channels == None and 'num_classes' in dir(val_dataset)) else None
    args.n_classes = args.model_config.output_channels[0]

    #################################################
    # create model
    print("=> creating model '{}'".format(args.model_name))

    model = onnx.load(args.pretrained)
    # Run the ONNX model with Caffe2
    onnx.checker.check_model(model)
    model = caffe2.python.onnx.backend.prepare(model)


    #################################################
    if args.palette:
        print('Creating palette')
        eval_string = args.palette
        palette = eval(eval_string)
        args.palette = np.zeros((256,3))
        for i, p in enumerate(palette):
            args.palette[i,0] = p[0]
            args.palette[i,1] = p[1]
            args.palette[i,2] = p[2]
        args.palette = args.palette[...,::-1] #RGB->BGR, since palette is expected to be given in RGB format

    infer_path = os.path.join(save_path, 'inference')
    if not os.path.exists(infer_path):
        os.makedirs(infer_path)

    #################################################
    with torch.no_grad():
        validate(args, val_dataset, val_loader, model, 0, infer_path)

    if args.create_video:
        create_video(args, infer_path=infer_path)

    if args.logger is not None:
        args.logger.close()
        args.logger = None


def validate(args, val_dataset, val_loader, model, epoch, infer_path):
    data_time = xnn.utils.AverageMeter()
    avg_metric = xnn.utils.AverageMeter()

    # switch to evaluate mode
    #model.eval()
    metric_name = "Metric"
    end_time = time.time()
    writer_idx = 0
    last_update_iter = -1
    metric_ctx = [None] * len(args.metric_modules)

    if args.label:
        confusion_matrix = np.zeros((args.n_classes, args.n_classes+1))
        for iter, (input_list, target_list, input_path, target_path) in enumerate(val_loader):
            data_time.update(time.time() - end_time)
            target_sizes = [tgt.shape for tgt in target_list]

            input_dict = {args.input_tensor_name[idx]: input_list[idx].numpy() for idx in range(len(input_list))}
            output = model.run(input_dict)[0]

            list_output = type(output) in (list, tuple)
            output_pred = output[0] if list_output else output

            if args.output_size is not None:
                output_pred = upsample_tensors(output_pred, target_sizes, args.upsample_mode)
            #
            if args.blend:
                for index in range(output_pred.shape[0]):
                    prediction = np.squeeze(np.array(output_pred[index]))
                    #prediction = np.argmax(prediction, axis = 0)
                    prediction_size = (prediction.shape[0], prediction.shape[1], 3)
                    output_image = args.palette[prediction.ravel()].reshape(prediction_size)
                    input_bgr = cv2.imread(input_path[0][index]) #Read the actual RGB image
                    input_bgr = cv2.resize(input_bgr, dsize=(prediction.shape[1],prediction.shape[0]))

                    output_image = chroma_blend(input_bgr, output_image)
                    output_name = os.path.join(infer_path, os.path.basename(input_path[0][index]))
                    cv2.imwrite(output_name, output_image)

            if args.label:
                for index in range(output_pred.shape[0]): 
                    prediction = np.array(output_pred[index])
                    #prediction = np.argmax(prediction, axis = 0)
                    label = np.squeeze(np.array(target_list[0][index]))
                    confusion_matrix = eval_output(args, prediction, label, confusion_matrix)
                    accuracy, mean_iou, iou, f1_score= compute_accuracy(args, confusion_matrix)
                print('pixel_accuracy={}, mean_iou={}, iou={}, f1_score = {}'.format(accuracy, mean_iou, iou, f1_score))
                sys.stdout.flush()
    else:
        for iter, (input_list, _ , input_path, _) in enumerate(val_loader):
            data_time.update(time.time() - end_time)

            input_dict = {args.input_tensor_name[idx]: input_list[idx].numpy() for idx in range(len(input_list))}
            output = model.run(input_dict)[0]

            list_output = type(output) in (list, tuple)
            output_pred = output[0] if list_output else output
            input_path = input_path[0]
            
            if args.output_size is not None:
                target_sizes = [args.output_size]
                output_pred = upsample_tensors(output_pred, target_sizes, args.upsample_mode)
            #
            if args.blend:
                for index in range(output_pred.shape[0]):
                    prediction = np.squeeze(np.array(output_pred[index])) #np.squeeze(np.array(output_pred[index].cpu()))
                    prediction_size = (prediction.shape[0], prediction.shape[1], 3)
                    output_image = args.palette[prediction.ravel()].reshape(prediction_size)
                    input_bgr = cv2.imread(input_path[index]) #Read the actual RGB image
                    input_bgr = cv2.resize(input_bgr, (args.img_resize[1], args.img_resize[0]), interpolation=cv2.INTER_LINEAR)
                    output_image = chroma_blend(input_bgr, output_image)
                    output_name = os.path.join(infer_path, os.path.basename(input_path[index]))
                    cv2.imwrite(output_name, output_image)
                    print('Inferred image {}'.format(input_path[index]))
            if args.car_mask:   #generating car_mask (required for localization)
                for index in range(output_pred.shape[0]):
                    prediction = np.array(output_pred[index])
                    prediction = np.argmax(prediction, axis = 0)
                    car_mask = np.logical_or(prediction == 13, prediction == 14, prediction == 16, prediction, prediction == 17)
                    prediction[car_mask] = 255
                    prediction[np.invert(car_mask)] = 0
                    output_image = prediction
                    output_name = os.path.join(infer_path, os.path.basename(input_path[index]))
                    cv2.imwrite(output_name, output_image)


def get_save_path(args, phase=None):
    date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join('./data/checkpoints/edgeailite', args.dataset_name, '{}_{}_'.format(date, args.model_name))
    save_path += 'b{}'.format(args.batch_size)
    phase = phase if (phase is not None) else args.phase
    save_path = os.path.join(save_path, phase)
    return save_path


def get_transforms(args):
    # image normalization can be at the beginning of transforms or at the end
    args.image_mean = np.array(args.image_mean, dtype=np.float32)
    args.image_scale = np.array(args.image_scale, dtype=np.float32)
    image_prenorm = xvision.transforms.image_transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) if args.image_prenorm else None
    image_postnorm = xvision.transforms.image_transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) if (not image_prenorm) else None

    #target size must be according to output_size. prediction will be resized to output_size before evaluation.
    test_transform = xvision.transforms.image_transforms.Compose([
        image_prenorm,
        xvision.transforms.image_transforms.AlignImages(),
        xvision.transforms.image_transforms.MaskTarget(args.target_mask, 0),
        xvision.transforms.image_transforms.CropRect(args.img_border_crop),
        xvision.transforms.image_transforms.Scale(args.img_resize, target_size=args.output_size, is_flow=args.is_flow),
        image_postnorm,
        xvision.transforms.image_transforms.ConvertToTensor()
        ])

    return test_transform


def _upsample_impl(tensor, output_size, upsample_mode):
    # upsample of long tensor is not supported currently. covert to float, just to avoid error.
    # we can do this only in the case of nearest mode, otherwise output will have invalid values.
    convert_tensor_to_float = False
    convert_np_to_float = False
    if isinstance(tensor, (torch.LongTensor,torch.cuda.LongTensor)):
        convert_tensor_to_float = True
        original_dtype = tensor.dtype
        tensor = tensor.float()
    elif isinstance(tensor, np.ndarray) and (np.dtype != np.float32):
        convert_np_to_float = True
        original_dtype = tensor.dtype
        tensor = tensor.astype(np.float32)
    #

    dim_added = False
    if len(tensor.shape) < 4:
        tensor = tensor[np.newaxis,...]
        dim_added = True
    #
    if (tensor.shape[-2:] != output_size):
        assert tensor.shape[1] == 1, 'TODO: add code for multi channel resizing'
        out_tensor = np.zeros((tensor.shape[0],tensor.shape[1],output_size[0],output_size[1]),dtype=np.float32)
        for b_idx in range(tensor.shape[0]):
            b_tensor = PIL.Image.fromarray(tensor[b_idx,0])
            b_tensor = b_tensor.resize((output_size[1],output_size[0]), PIL.Image.NEAREST)
            out_tensor[b_idx,0,...] = np.asarray(b_tensor)
        #
        tensor = out_tensor
    #
    if dim_added:
        tensor = tensor[0]
    #

    if convert_tensor_to_float:
        tensor = tensor.long()
    elif convert_np_to_float:
        tensor = tensor.astype(original_dtype)
    #
    return tensor

def upsample_tensors(tensors, output_sizes, upsample_mode):
    if isinstance(tensors, (list,tuple)):
        for tidx, tensor in enumerate(tensors):
            tensors[tidx] = _upsample_impl(tensor, output_sizes[tidx][-2:], upsample_mode)
        #
    else:
        tensors = _upsample_impl(tensors, output_sizes[0][-2:], upsample_mode)
    return tensors


def chroma_blend(image, color):
    image_yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YUV)
    image_y,image_u,image_v = cv2.split(image_yuv)
    color_yuv = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_BGR2YUV)
    color_y,color_u,color_v = cv2.split(color_yuv)
    image_y = np.uint8(image_y)
    color_u = np.uint8(color_u)
    color_v = np.uint8(color_v)
    image_yuv = cv2.merge((image_y,color_u,color_v))
    image = cv2.cvtColor(image_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    return image    



def eval_output(args, output, label, confusion_matrix):

    if len(label.shape)>2:
        label = label[:,:,0]
    gt_labels = label.ravel()
    det_labels = output.ravel().clip(0,args.n_classes)
    gt_labels_valid_ind = np.where(gt_labels != 255)
    gt_labels_valid = gt_labels[gt_labels_valid_ind]
    det_labels_valid = det_labels[gt_labels_valid_ind]
    for r in range(confusion_matrix.shape[0]):
        for c in range(confusion_matrix.shape[1]):
            confusion_matrix[r,c] += np.sum((gt_labels_valid==r) & (det_labels_valid==c))

    return confusion_matrix
    
def compute_accuracy(args, confusion_matrix):

    #pdb.set_trace()
    num_selected_classes = args.n_classes
    tp = np.zeros(args.n_classes)
    population = np.zeros(args.n_classes)
    det = np.zeros(args.n_classes)
    iou = np.zeros(args.n_classes)
    
    for r in range(args.n_classes):
      for c in range(args.n_classes):   
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
          
    mean_iou = np.sum(iou) / num_nonempty_classes
    accuracy = np.sum(tp) / np.sum(population)
    
    #F1 score calculation
    fp = np.zeros(args.n_classes)
    fn = np.zeros(args.n_classes)
    precision = np.zeros(args.n_classes)
    recall = np.zeros(args.n_classes)
    f1_score = np.zeros(args.n_classes)

    for cls in range(num_selected_classes):
        fp[cls] = det[cls] - tp[cls]
        fn[cls] = population[cls] - tp[cls]
        precision[cls] = tp[cls] / (det[cls] + 1e-10)
        recall[cls] = tp[cls] / (tp[cls] + fn[cls] + 1e-10)        
        f1_score[cls] = 2 * precision[cls]*recall[cls] / (precision[cls] + recall[cls] + 1e-10)

    return accuracy, mean_iou, iou, f1_score
    
        
def infer_video(args, net):
    videoIpHandle = imageio.get_reader(args.input, 'ffmpeg')
    fps = math.ceil(videoIpHandle.get_meta_data()['fps'])
    print(videoIpHandle.get_meta_data())
    numFrames = min(len(videoIpHandle), args.num_images)
    videoOpHandle = imageio.get_writer(args.output,'ffmpeg', fps=fps)
    for num in range(numFrames):
        print(num, end=' ')
        sys.stdout.flush()
        input_blob = videoIpHandle.get_data(num)
        input_blob = input_blob[...,::-1]    #RGB->BGR
        output_blob = infer_blob(args, net, input_blob)     
        output_blob = output_blob[...,::-1]  #BGR->RGB            
        videoOpHandle.append_data(output_blob)
    videoOpHandle.close()
    return

def create_video(args, infer_path):
    op_file_name = args.data_path.split('/')[-1]
    os.system(' ffmpeg -framerate 30 -pattern_type glob -i "{}/*.png" -c:v libx264 -vb 50000k -qmax 20 -r 10 -vf \
                 scale=1024:512  -pix_fmt yuv420p {}.MP4'.format(infer_path, op_file_name))

if __name__ == '__main__':
    train_args = get_config()
    train_args = parser.parse_args()
    main(train_args)
