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

import copy
import numpy as np


def dict_update(src_dict, *args, inplace=False, **kwargs):
    new_dict = src_dict if inplace else copy.deepcopy(src_dict)
    for arg in args:
        assert isinstance(arg, dict), 'arguments must be dict or keywords'
        new_dict.update(arg)
    #
    new_dict.update(kwargs)
    return new_dict


def dict_merge(target_dict, src_dict, inplace=False):
    target_dict = target_dict if inplace else copy.deepcopy(target_dict)
    assert isinstance(target_dict, dict), 'destination must be a dict'
    assert isinstance(src_dict, dict), 'source must be a dict'
    for key, value in src_dict.items():
        if hasattr(target_dict, key) and isinstance(target_dict[key], dict):
            if isinstance(value, dict):
                target_dict[key] = dict_merge(target_dict[key], **value)
            else:
                target_dict[key] = value
            #
        else:
            target_dict[key] = value
        #
    #
    return target_dict


def dict_equal(self, shape1, shape2):
    for k1, v1 in shape1.items():
        if k1 not in shape2:
            return False
        #
        v2 = shape2[k1]
        if isinstance(v1, (list,tuple)) or isinstance(v2, (list,tuple)):
            if any(v1 != v2):
                return False
            #
        elif v1 != v2:
            return False
        #
    #
    return True


def as_tuple(arg):
    return arg if isinstance(arg, tuple) else (arg,)


def as_list(arg):
    return arg if isinstance(arg, list) else [arg]


def as_list_or_tuple(arg):
    return arg if isinstance(arg, (list,tuple)) else (arg,)


def round_dict(d, precision=3):
    d_copy = {}
    for k, v in d.items():
        # numpy objects cannot be serialized with yaml - convert to float
        if isinstance(v, (np.float32, np.float64)):
            v = float(v)
        #
        # round to the given precision
        if isinstance(v, float):
            v = round(v, precision)
        #
        d_copy.update({k:v})
    #
    return d_copy


def round_dicts(d_in, precision=3):
    if isinstance(d_in, (list,tuple)):
        r_out = []
        for d_idx, d in enumerate(d_in):
            d = round_dict(d, precision) if isinstance(d, dict) else d
            r_out.append(d)
        #
    elif isinstance(d_in, dict):
        r_out = {}
        for d_key, d in d_in.items():
            d = round_dict(d, precision) if isinstance(d, dict) else d
            r_out.update({d_key:d})
        #
    else:
        r_out = d_in
    #
    return r_out
