#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import os
import random
import copy
import collections
import threading
import PIL

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

basename = os.path.splitext(os.path.basename(__file__))[0]
if __name__.startswith(basename):
    import transforms
else:
    from . import transforms
#


class RandomMosaic():
    def __init__(self, augmentations=None, prob=0.75,  max_crops=4, regular_mosaic=True):
        ''' Mosaic data augmentation - efficient implementation using history.
            Recommended (not mandatory, to give teh crop_size argument)
        '''
        assert prob >= 0.0 and prob <= 1.0, 'prob must be within 0.0 to 1.0'
        super().__init__()
        self.prob = prob
        self.max_crops = max_crops
        self.regular_mosaic = regular_mosaic
        self.augmentations = augmentations
        # maxlen is set to 5 to include 4 past images and the current image
        # this helps us to pick the current image with the probability prob
        self.image_history = collections.deque(maxlen=5)
        self.target_history = collections.deque(maxlen=5)

    def __call__(self, image_tensor, target_tensor):
        with torch.no_grad():
            return self.forward_transform(image_tensor, target_tensor)

    def forward_transform(self, image_tensor, target_tensor):
        # save the current image and target in history
        self.put_in_history(image_tensor, target_tensor)
        # create a template output with current tensor by applying augmentations
        image_tensor, target_tensor = self.augmentations(image_tensor, target_tensor)
        # now do the actual transform
        # crops can be random mosaic or completely arbitrary
        crops = self.get_mosaic_crops(image_tensor) if self.regular_mosaic else \
            self.get_rand_crops(image_tensor)
        # loop over crops
        for crop in crops:
            # get the source image to get crop from and apply augmentations
            image_src, target_src = self.get_from_history()
            image_src, target_src = self.augmentations(image_src, target_src)
            # now actually copy the crop
            self.copy_crop(crop, image_tensor, target_tensor, image_src, target_src)
        #
        return image_tensor, target_tensor

    def put_in_history(self, image_tensor, target_tensor):
        self.image_history.append(image_tensor)
        self.target_history.append(target_tensor)

    def get_from_history(self):
        history_len = len(self.image_history)
        if history_len >= 2 and random.random() < self.prob:
            hist_index = random.randint(0, history_len-2)
        else:
            hist_index = -1
        #
        image_src = self.image_history[hist_index]
        target_src = self.target_history[hist_index]
        return image_src, target_src

    def copy_crop(self, crop_dst, image_tensor, target_tensor, image_src, target_src):
        crop_r1, crop_c1, crop_r2, crop_c2 = crop_dst[0], crop_dst[1], crop_dst[0]+crop_dst[2], crop_dst[1]+crop_dst[3]
        if image_src.shape[-2] > crop_r1 and image_src.shape[-2] > crop_r2 and \
            image_src.shape[-1] > crop_c1 and image_src.shape[-1] > crop_c2 and \
            image_tensor.shape[-2] > crop_r1 and image_tensor.shape[-2] > crop_r2 and \
            image_tensor.shape[-1] > crop_c1 and image_tensor.shape[-1] > crop_c2:
            image_tensor[:, crop_r1:crop_r2, crop_c1:crop_c2] = image_src[:, crop_r1:crop_r2, crop_c1:crop_c2].detach()
            if "boxes" in target_src:
                target_tensor["boxes"] = torch.cat((target_tensor["boxes"], target_src["boxes"]), dim=0).detach()
            if "labels" in target_src:
                target_tensor["labels"] = torch.cat((target_tensor["labels"], target_src["labels"]), dim=0).detach()
            if "keypoints" in target_src:
                target_tensor["keypoints"] = torch.cat((target_tensor["keypoints"], target_src["keypoints"]), dim=0).detach()
            # if "masks" in target_src:
            #     target_tensor["masks"] = torch.cat((target_tensor["masks"], target_src["masks"]), dim=0)
        #
        return

    def get_mosaic_crops(self, image_tensor):
        w, h = image_tensor.size()[-2:]
        crop_pos = (random.randint(2*h//10, 8*h//10), random.randint(2*w//10, 8*w//10))
        crops = [[0, 0, crop_pos[0]-0, crop_pos[1]-0],
                 [0, crop_pos[1], crop_pos[0]-0, w-crop_pos[1]],
                 [crop_pos[0], 0, h-crop_pos[0], crop_pos[1]-0],
                 [crop_pos[0], crop_pos[1], h-crop_pos[0], w-crop_pos[1]]]
        return crops

    def get_rand_crops(self, image_tensor):
        crops = []
        w, h = image_tensor.size()[-2:]
        for _ in range(self.max_crops):
            crop_size = (random.randint(2*h//10, 8*h//10), random.randint(2*w//10, 8*w//10))
            crop_params = T.RandomCrop.get_params(image_tensor, crop_size)
            crops.append(crop_params)
        #
        return crops
