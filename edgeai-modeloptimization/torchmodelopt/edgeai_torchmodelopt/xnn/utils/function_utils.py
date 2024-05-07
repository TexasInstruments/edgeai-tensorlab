#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
import copy
import functools
import torch


#########################################################################
# a utility function used for argument parsing
def str2bool(v):
  if isinstance(v, (str)):
      if v.lower() in ("yes", "true", "t", "1"):
          return True
      elif v.lower() in ("no", "false", "f", "0"):
          return False
      else:
          return v
      #
  else:
      return v

def splitstr2bool(v):
  v = v.split(',')
  for index, args in enumerate(v):
      v[index] = str2bool(args)
  return v


#########################################################################
def make_divisible(value, factor, min_value=None):
    """
    Inspired by https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    min_value = factor if min_value is None else min_value
    round_factor = factor/2
    quotient = int(value + round_factor) // factor
    value_multiple = max(quotient * factor, min_value)
    # make sure that the change is contained
    if value_multiple < 0.9*value:
        value_multiple = value_multiple + factor
    #
    return int(value_multiple)


def make_divisible_by8(v):
    return make_divisible(v, 8)


#########################################################################
def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


#########################################################################
def get_shape_with_stride(in_shape, stride):
    shape_s = [in_shape[0],in_shape[1],in_shape[2]//stride,in_shape[3]//stride]
    if (int(in_shape[2]) % 2) == 1:
        shape_s[2] += 1
    if (int(in_shape[3]) % 2) == 1:
        shape_s[3] += 1
    return shape_s


def get_blob_from_list(x_list, search_shape, start_dim=None):
    x_ret = None
    start_dim = start_dim if start_dim is not None else 0
    for x in x_list:
        if isinstance(x, list):
            x = torch.cat(x,dim=1)
        #
        if (x.shape[start_dim:] == torch.Size(search_shape[start_dim:])):
            x_ret = x
        #
    return x_ret


#########################################################################
def partialclass(cls, *args, class_name='PartialClass', **kwargs):
    __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    PartialClass = type(class_name, (cls,), {'__init__': __init__})
    return PartialClass


# Use the more compact implementation above
# def partialclass(cls, *new_args, class_name='PartialClass', **new_kwargs):
#     def __init__(self, *args, **kwargs):
#         kwargs = copy.deepcopy(kwargs)
#         for k in new_kwargs:
#             if k in kwargs:
#                 kwargs.pop(k)
#             #
#         #
#         cls.__init__(self, *args, *new_args, **kwargs, **new_kwargs)
#     #
#     PartialClass = type(class_name, (cls,), {'__init__': __init__})
#     return PartialClass


# This implementation works, but better to use one of the above implementation to specify class_name as well
# def partialclass(cls, *args, **kwargs):
#     class PartialClass(cls):
#         __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
#     #
#     return PartialClass

#########################################################################
