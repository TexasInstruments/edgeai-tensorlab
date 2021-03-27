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

from .coco_det import *
from .coco_seg import *
from .imagenet import *
from .imagenet_subset import *
from .cityscapes import *
from .ade20k import *
from .voc_seg import *


imagenetcls_dataset_type_dict = {'imagenet':ImageNetCls,
                                 'tiny-imagenet200':TinyImageNet200Cls,
                                 'imagenet-dogs120':ImageNetDogs120Cls,
                                 'imagenet-psuedo120':ImageNetPseudo120Cls,
                                 'imagenet-resized-64x64':ImageNetResized64x64Cls}

imagecls_dataset_type_dict = {'generic':ImageCls}.update(imagenetcls_dataset_type_dict)


imagenetcls_dataset_size_dict = {'imagenet':50000,
                                 'tiny-imagenet200':10000,
                                 'imagenet-dogs120':20580,
                                 'imagenet-psuedo120':20580,
                                 'imagenet-resized-64x64':50000}

imagecls_dataset_size_dict = {'generic':None}.update(imagenetcls_dataset_size_dict)


# imagenet-dogs120 doesn't have val data
imagenetcls_dataset_splits_dict = {'imagenet':['train','val'],
                                   'tiny-imagenet200':['train','val'],
                                   'imagenet-dogs120':['train','train'],
                                   'imagenet-psuedo120':['train','train'],
                                   'imagenet-resized-64x64':['train','val']}


imagecls_dataset_splits_dict = {'generic':['train','val']}.update(imagenetcls_dataset_splits_dict)
