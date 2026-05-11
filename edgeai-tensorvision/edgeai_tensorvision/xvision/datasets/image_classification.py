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
from torchvision.datasets import folder
from torchvision.datasets import cifar
from torchvision.datasets import imagenet

__all__ = ['image_folder_classification_train', 'image_folder_classification_validation', 'image_folder_classification',
           'imagenet_classification_train', 'imagenet_classification_validation', 'imagenet_classification',
           'cifar10_classification', 'cifar100_classification']

########################################################################
def image_folder_classification_train(dataset_config, root, split=None, transforms=None):
    split = 'train' if split is None else split
    traindir = os.path.join(root, split)
    assert os.path.exists(traindir), f'dataset training folder does not exist {traindir}'
    train_transform = transforms[0] if isinstance(transforms,(list,tuple)) else transforms
    train_dataset = folder.ImageFolder(traindir, train_transform)
    return train_dataset

def image_folder_classification_validation(dataset_config, root, split=None, transforms=None):
    split = 'val' if split is None else split
    # validation folder can be either 'val' or 'validation'
    if (split == 'val') and (not os.path.exists(os.path.join(root,split))):
        split = 'validation'
    #
    valdir = os.path.join(root, split)
    assert os.path.exists(valdir), f'dataset validation folder does not exist {valdir}'
    val_transform = transforms[1] if isinstance(transforms,(list,tuple)) else transforms
    val_dataset = folder.ImageFolder(valdir, val_transform)
    return val_dataset

def image_folder_classification(dataset_config, root, split=None, transforms=None):
    split = ('train', 'val') if split is None else split
    train_transform, val_transform = transforms
    train_dataset = image_folder_classification_train(dataset_config, root, split[0], train_transform)
    val_dataset = image_folder_classification_validation(dataset_config, root, split[1], val_transform)
    return train_dataset, val_dataset

########################################################################
def imagenet_classification_train(dataset_config, root, split=None, transforms=None):
    train_transform = transforms[0] if isinstance(transforms,(list,tuple)) else transforms
    train_dataset = imagenet.ImageNet(root, train=True, transform=train_transform, target_transform=None, download=True)
    return train_dataset

def imagenet_classification_validation(dataset_config, root, split=None, transforms=None):
    val_transform = transforms[1] if isinstance(transforms,(list,tuple)) else transforms
    val_dataset = imagenet.ImageNet(root, train=False, transform=val_transform, target_transform=None, download=True)
    return val_dataset

def imagenet_classification(dataset_config, root, split=None, transforms=None):
    train_dataset = imagenet_classification_train(dataset_config, root, split, transforms)
    val_dataset = imagenet_classification_validation(dataset_config, root, split, transforms)
    return train_dataset, val_dataset


########################################################################
def cifar10_classification(dataset_config, root, split=None, transforms=None):
    train_dataset = cifar.CIFAR10(root, train=True, transform=transforms[0], target_transform=None, download=True)
    val_dataset = cifar.CIFAR10(root, train=False, transform=transforms[1], target_transform=None, download=True)
    return train_dataset, val_dataset

def cifar100_classification(dataset_config, root, split=None, transforms=None):
    train_dataset = cifar.CIFAR100(root, train=True, transform=transforms[0], target_transform=None, download=True)
    val_dataset = cifar.CIFAR100(root, train=False, transform=transforms[1], target_transform=None, download=True)
    return train_dataset, val_dataset