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

################################################################################

# Also includes parts from: https://github.com/pytorch/vision
# License: https://github.com/pytorch/vision/blob/master/LICENSE
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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


import numbers
from collections.abc import Sequence
import numpy as np
import PIL
import cv2

from PIL import Image
from . import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class ImageRead(object):
    def __init__(self, backend='pil'):
        assert backend in ('pil', 'cv2'), f'backend must be one of pil or cv2. got {backend}'
        self.backend = backend

    def __call__(self, path, info_dict):
        if isinstance(path, str):
            img_data = None
            if self.backend == 'pil':
                img_data = PIL.Image.open(path)
                img_data = img_data.convert('RGB')
                info_dict['data_shape'] = img_data.size[1], img_data.size[0], len(img_data.getbands())
            elif self.backend == 'cv2':
                img_data = cv2.imread(path)
                if img_data.shape[-1] == 1:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                elif img_data.shape[-1] == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                #
                # always return in RGB format
                img_data = img_data[:,:,::-1]
                info_dict['data_shape'] = img_data.shape
            #
            info_dict['data'] = img_data
            info_dict['data_path'] = path
        elif isinstance(path, np.ndarray):
            img_data = path
            info_dict['data_shape'] = img_data.shape
            info_dict['data'] = img_data
            info_dict['data_path'] = './'
        elif isinstance(path, tuple):
            img_data = []
            for i in range(len(path)):
                if self.backend == 'pil':
                    img_data.append(PIL.Image.open(path[i]))
                    img_data[i] = img_data[i].convert('RGB')
                    info_dict['data_shape'] = img_data.size[1], img_data.size[0], len(img_data.getbands())
                elif self.backend == 'cv2':
                    img_data.append(cv2.imread(path[i]))
                    if img_data[i].shape[-1] == 1:
                        img_data[i] = cv2.cvtColor(img_data[i], cv2.COLOR_GRAY2BGR)
                    elif img_data[i].shape[-1] == 4:
                        img_data[i] = cv2.cvtColor(img_data[i], cv2.COLOR_BGRA2BGR)
                    #
                    # always return in RGB format
                    img_data[i] = img_data[i][:,:,::-1]

            if self.backend == 'pil':
                info_dict['data_shape'] = img_data[0].size[1], img_data[0].size[0], len(img_data[0].getbands())
            else:
                info_dict['data_shape'] = img_data[0].shape            
        else:
            assert False, 'invalid input'
        #
        return img_data, info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'(backend={self.backend})'


class ImageNorm(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, data_layout='NCHW', inplace=False):
        self.mean = mean
        self.std = std
        self.data_layout = data_layout
        self.inplace = inplace

    def __call__(self, tensor, info_dict):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, (list,tuple)):
            tensor = [F.normalize(t, self.mean, self.std, self.data_layout, self.inplace) for t in tensor]
        else:
            tensor = F.normalize(tensor, self.mean, self.std, self.data_layout, self.inplace)
        #
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ImageNormMeanScale(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and scale: ``(scale[1],..,scale[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``output[channel] = (input[channel] - mean[channel]) * scale[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        scale (sequence): Sequence of factors for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, scale, data_layout='NCHW', inplace=False):
        self.mean = mean
        self.scale = scale
        self.data_layout = data_layout
        self.inplace = inplace

    def __call__(self, tensor, info_dict):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, (list,tuple)):
            tensor = [F.normalize_mean_scale(t, self.mean, self.scale, self.data_layout, self.inplace) for t in tensor]
        elif isinstance(tensor, dict):
            tensor = {name:F.normalize_mean_scale(t, self.mean, self.scale, self.data_layout, self.inplace) for name, t in tensor.items()}
        else:
            tensor = F.normalize_mean_scale(tensor, self.mean, self.scale, self.data_layout, self.inplace)
        #
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, scale={1})'.format(self.mean, self.scale)


class ImageResize():
    """Resize the input image to the given size.
    The image can be a PIL Image, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, *args, **kwargs):
        super().__init__()
        self.size = size
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, list):
            for i in range(len(img)):
                img[i], border = F.resize(img[i], self.size, *self.args, **self.kwargs)
            
            if isinstance(img[0], np.ndarray):
                info_dict['resize_shape'] = img[0].shape
            else:
                info_dict['resize_shape'] = img[0].size[1],  img[0].size[0], len(img[0].getbands())
            #
            info_dict['resize_border'] = border

        else:        
            img, border = F.resize(img, self.size, *self.args, **self.kwargs)
            if isinstance(img, np.ndarray):
                info_dict['resize_shape'] = img.shape
            else:
                info_dict['resize_shape'] = img.size[1], img.size[0], len(img.getbands())
            #
            info_dict['resize_border'] = border

        return img, info_dict

    def set_size(self, size):
        self.size = size

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'({self.size}'
        for arg in self.args:
            repr_str += f', {arg}'
        #
        for k, v in self.kwargs.items():
            repr_str += f', {k}={v}'
        #
        repr_str += ')'
        return repr_str


class ImageCenterCrop():
    """Crops the given image at the center.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size=None):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        elif size is None:
            self.size = size
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size = size
        #

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is not None:
            if isinstance(img, list):
                for i in range(len(img)):
                    img[i] = F.center_crop(img[i], self.size)          
            else:        
                img = F.center_crop(img, self.size)            

        return img, info_dict
        #return F.center_crop(img, self.size) if self.size is not None else img, info_dict

    def set_size(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        elif size is None:
            self.size = size
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size = size
        #

    def __repr__(self):
        if self.size is not None:
            return self.__class__.__name__ + '(size={0})'.format(self.size)
        else:
            return self.__class__.__name__ + '()'


class ImageToNPTensor(object):
    """Convert a ``Image`` to a tensor of the same type.

    Converts a PIL Image or numpy array (H x W x C) to a numpy Tensor of shape (C x H x W).
    """

    def __init__(self, data_layout='NCHW', reverse_channels=False):
        self.data_layout = data_layout
        self.reverse_channels = reverse_channels

    def __call__(self, pic, info_dict):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_numpy_tensor(pic, self.data_layout, self.reverse_channels), info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout}, {self.reverse_channels})'


class ImageToNPTensor4D(object):
    """Convert a ``Image`` to a tensor of the same type.

    Converts a PIL Image or numpy array (H x W x C) to a numpy Tensor of shape (C x H x W).
    """

    def __init__(self, data_layout='NCHW', reverse_channels=False):
        self.data_layout = data_layout
        self.reverse_channels = reverse_channels

    def __call__(self, pic, info_dict):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, list):
            for i in range(len(pic)):
                pic[i] = F.to_numpy_tensor_4d(pic[i], self.data_layout, self.reverse_channels)                
        else:        
            pic = F.to_numpy_tensor_4d(pic, self.data_layout, self.reverse_channels)        

        return pic, info_dict
        #return F.to_numpy_tensor_4d(pic, self.data_layout, self.reverse_channels), info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout}, {self.reverse_channels})'


class NPTensor4DChanReverse(object):
    """Convert a ``Image`` to a tensor of the same type.

    Converts a PIL Image or numpy array (H x W x C) to a numpy Tensor of shape (C x H x W).
    """

    def __init__(self, data_layout='NCHW'):
        assert data_layout in ('NCHW', 'NHWC'), f'invalid data_layout {data_layout}'
        self.data_layout = data_layout

    def __call__(self, pic, info_dict):
        """
        Args:
            pic (np.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted tensor.
        """
        if self.data_layout == 'NCHW':
            return pic[:,::-1,...], info_dict
        else:
            return pic[...,::-1], info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout})'

class ImageFlipAdd:
    def __init__(self, flip_axis = 3):
        self.flip_axis = flip_axis

    def __call__(self, img, info_dict):
        info_dict['flip_img'] = np.flip(img, axis=[self.flip_axis])
        return img, info_dict

class PointCloudRead(object):
    def __init__(self):
        pass

    def __call__(self, path, info_dict):
        point_cloud_data = None

        point_cloud_data = np.fromfile(path, dtype='float32')
        point_cloud_data = np.reshape(point_cloud_data,(-1,4))

        info_dict['data_shape'] = point_cloud_data.shape[1], point_cloud_data.shape[0]
        info_dict['data'] = point_cloud_data
        info_dict['data_path'] = path

        return point_cloud_data, info_dict

class Voxelization(object):
    def __init__(self):

        self.min_x = 0
        self.max_x = 69.120
        self.min_y = -39.680
        self.max_y = 39.680
        self.min_z = -3.0
        self.max_z = 1.0
        self.voxel_size_x= 0.16
        self.voxel_size_y= 0.16
        self.voxel_size_z= 4.0
        self.num_voxel_x = (self.max_x - self.min_x)/self.voxel_size_x
        self.num_voxel_y = (self.max_y - self.min_y)/self.voxel_size_y
        self.max_points_per_voxel = 32
        self.nw_max_num_voxels  = 10000
        self.num_feat_per_voxel = 10
        self.num_channel = 64
        self.scale_fact = 32.0


    def __call__(self, lidar_data, info_dict):

        scratch_2 =[]
        enable_pre_proc = True
        enable_opt_pre_proc = True

        input1 = np.zeros((1, self.num_channel, (int)(self.num_voxel_x*self.num_voxel_y)),dtype='float32')

        if enable_pre_proc == True:
            input0 = np.zeros((1, self.num_feat_per_voxel, self.max_points_per_voxel, self.nw_max_num_voxels),dtype='float32')
            input2 = np.zeros((1, self.num_channel, self.nw_max_num_voxels),dtype='int32')

            #start_time = time.time()

            if enable_opt_pre_proc == False:

                scratch_1 = []
                for i, data in enumerate(lidar_data):

                    x = data[0]
                    y = data[1]
                    z = data[2]

                    if ((x > self.min_x) and (x < self.max_x) and (y > self.min_y) and (y < self.max_y) and
                        (z > self.min_z) and (z < self.max_z)):

                        x_id = (((x - self.min_x) / self.voxel_size_x)).astype(int)
                        y_id = (((y - self.min_y) / self.voxel_size_y)).astype(int)
                        scratch_1.append(y_id * self.num_voxel_x + x_id)
                    else:
                        scratch_1.append(-1 - i) # filing unique non valid index
            else:
                x = lidar_data[:, 0]
                y = lidar_data[:, 1]
                z = lidar_data[:, 2]

                x_id = ((((x - self.min_x) / self.voxel_size_x))).astype(int)
                y_id =  ((((y - self.min_y) / self.voxel_size_y))).astype(int)
                valid_idx = y_id * self.num_voxel_x + x_id
                not_valid_idx = np.ones(len(lidar_data))*-1 # -1  is the invalid index

                valid_pts = (x > self.min_x)*(x < self.max_x)*(y > self.min_y)*\
                            (y < self.max_y)*(z > self.min_z)*(z < self.max_z)

                indx_write = np.where(valid_pts,valid_idx,not_valid_idx)

                scratch_1  = indx_write

            num_points = np.zeros(self.nw_max_num_voxels,dtype=int)

            # Find unique indices
            # There will be voxel which doesnt have any 3d point, hence collecting the voxel ids for valid voxels*/
            # scratch_2 is the index in valid voxels
            num_non_empty_voxels = 0

            lidar_data = lidar_data[np.where(scratch_1 != -1)]
            scratch_1 = scratch_1[np.where(scratch_1 != -1)]

            if enable_opt_pre_proc == False:
                for i in range(len(lidar_data)):
                    if (scratch_1[i] >= 0):

                        find_voxel = scratch_1[i] in scratch_1[:i]

                        if find_voxel == False:
                            scratch_2.append(num_non_empty_voxels) # this voxel idx has come first time, hence allocate a new index for this
                            input2[0][0][num_non_empty_voxels] = scratch_1[i]
                            num_non_empty_voxels += 1
                        else:
                            if enable_opt_pre_proc == False:
                                k = scratch_1[:i].index(scratch_1[i])
                            else:
                                k = (np.where(scratch_1[:i] == scratch_1[i]))[0][0]
                            scratch_2.append(scratch_2[k]) #already this voxel is having one id hence reuse it
                    else:
                        scratch_2.append(None)
            else:
                unq_rtn = np.unique(scratch_1, return_inverse = True, return_counts = True)
                num_non_empty_voxels = (int)(unq_rtn[2].shape[0])
                #num_points = unq_rtn[2]
                #num_points = np.clip(num_points,0,self.max_points_per_voxel)
                scratch_2 = unq_rtn[1]
                input2[0, 0, :num_non_empty_voxels] = unq_rtn[0][:num_non_empty_voxels]


            #Even though current_voxels is less than self.nw_max_num_voxels, then also arrange
            #    the data as per maximum number of voxels.

            if enable_opt_pre_proc == False:
                tot_num_pts = 0
                for i in range(len(lidar_data)):
                    if (scratch_1[i] >= 0):
                        j = scratch_2[i] #voxel index
                        if(num_points[j]<self.max_points_per_voxel):
                            input0[0, 0:4, num_points[j], j] = lidar_data[i, 0:4] * self.scale_fact
                            num_points[j] = num_points[j] + 1
                        else:
                            tot_num_pts = tot_num_pts+1
            else:
                for i in range(len(lidar_data)):
                    j = scratch_2[i] #voxel index
                    if(num_points[j] < self.max_points_per_voxel):
                        input0[0, 0:4, num_points[j], j] = lidar_data[i, 0:4] * self.scale_fact
                        num_points[j] = num_points[j] + 1

            line_pitch = self.nw_max_num_voxels
            channel_pitch = self.max_points_per_voxel * line_pitch
            x_offset = self.voxel_size_x / 2 + self.min_x
            y_offset = self.voxel_size_y / 2 + self.min_y
            z_offset = self.voxel_size_z / 2 + self.min_z

            for i in range(num_non_empty_voxels):
                x = 0
                y = 0
                z = 0

                if enable_opt_pre_proc == False:
                    for j in range(num_points[i]):
                        x += input0[0][0][j][i]
                        y += input0[0][1][j][i]
                        z += input0[0][2][j][i]
                else:
                    x = input0[0, 0, :num_points[i], i].sum()
                    y = input0[0, 1, :num_points[i], i].sum()
                    z = input0[0, 2, :num_points[i], i].sum()

                x_avg = x / num_points[i]
                y_avg = y / num_points[i]
                z_avg = z / num_points[i]

                voxel_center_y = (int)(input2[0][0][i] / self.num_voxel_x)
                voxel_center_x = (int)(input2[0][0][i] - ((int)(voxel_center_y)) * self.num_voxel_x)

                voxel_center_x *= self.voxel_size_x
                voxel_center_x += x_offset

                voxel_center_y *= self.voxel_size_y
                voxel_center_y += y_offset

                voxel_center_z = 0
                voxel_center_z *= self.voxel_size_z
                voxel_center_z += z_offset
                if enable_opt_pre_proc == False:
                    for j in range(num_points[i]):
                        input0[0][4][j][i] = input0[0][0][j][i] - x_avg
                        input0[0][5][j][i] = input0[0][1][j][i] - y_avg
                        input0[0][6][j][i] = input0[0][2][j][i] - z_avg
                        input0[0][7][j][i] = input0[0][0][j][i] - voxel_center_x * self.scale_fact
                        input0[0][8][j][i] = input0[0][1][j][i] - voxel_center_y * self.scale_fact
                        input0[0][9][j][i] = input0[0][2][j][i] - voxel_center_z * self.scale_fact

                    #/*looks like bug in python mmdetection3d code, hence below code is to mimic the mmdetect behaviour*/
                    for j in range (num_points[i]):
                        input0[0][0][j][i] = input0[0][7][j][i]
                        input0[0][1][j][i] = input0[0][8][j][i]
                        input0[0][2][j][i] = input0[0][9][j][i]
                else:
                    input0[0, 4, :num_points[i], i] = input0[0, 0, :num_points[i], i] - x_avg
                    input0[0, 5, :num_points[i], i] = input0[0, 1, :num_points[i], i] - y_avg
                    input0[0, 6, :num_points[i], i] = input0[0, 2, :num_points[i], i] - z_avg
                    input0[0, 7, :num_points[i], i] = input0[0, 0, :num_points[i], i] - voxel_center_x * self.scale_fact
                    input0[0, 8, :num_points[i], i] = input0[0, 1, :num_points[i], i] - voxel_center_y * self.scale_fact
                    input0[0, 9, :num_points[i], i] = input0[0, 2, :num_points[i], i] - voxel_center_z * self.scale_fact

                    input0[0, 0, :num_points[i], i] = input0[0, 7, :num_points[i], i]
                    input0[0, 1, :num_points[i], i] = input0[0, 8, :num_points[i], i]
                    input0[0, 2, :num_points[i], i] = input0[0, 9, :num_points[i], i]


            input2[0][0][num_non_empty_voxels] = -1 # TIDL doesnt know valid number of voxels, hence this act as marker field.
            input2[0][1:64] = input2[0][0] # replicating the firsh channel indices to all channels. As scatter is same for all channels.
            input0 = input0.astype("int32")
            input0 = input0.astype("float32")
            #input2 = input2.astype("float32")
            #np.savetxt('input2.txt', input2.flatten(), fmt='%6.2e')
            #np.savetxt('input0.txt', input0.flatten(), fmt='%6.2e')
        else:
            input0 = np.fromfile(info_dict['data_path'] + "_input0_f32.bin", dtype='float32')
            input2 = np.fromfile(info_dict['data_path'] + "_input2_f32.bin", dtype='float32')

            #np.savetxt('input2.txt', input2.flatten(), fmt='%6.2e')
            #np.savetxt('input0.txt', input0.flatten(), fmt='%6.2e')

            input0 = input0.astype("int32")
            input0 = input0.astype("float32")

            input0 = input0.reshape(1, 9, 32, 10000)
            input2 = input2.reshape(1, 64, 10000).astype('int32')

        return (input0,input2,input1), info_dict
