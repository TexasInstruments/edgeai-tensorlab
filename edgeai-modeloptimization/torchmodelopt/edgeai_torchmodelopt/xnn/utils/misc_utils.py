# Copyright (c) 2018-2023, Texas Instruments
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
import numpy as np
import yaml
from .params_base import ParamsBase


def dict_update(src_dict, *args, inplace=False, **kwargs):
    new_dict = src_dict if inplace else src_dict.copy()
    for arg in args:
        assert isinstance(arg, dict), 'arguments must be dict or keywords'
        new_dict.update(arg)
    #
    new_dict.update(kwargs)
    return new_dict


def dict_update_cond(src_dict, *args, inplace=False, condition_fn=None, **kwargs):
    condition_fn = condition_fn if condition_fn is not None else lambda x: (x is not None)
    def _update_conditional(new_dict, arg):
        conditional_arg = {k: v for k,v in arg.items() if condition_fn(v)}
        new_dict.update(conditional_arg)
    #
    new_dict = src_dict if inplace else src_dict.copy()
    for arg in args:
        assert isinstance(arg, dict), 'arguments must be dict or keywords'
        _update_conditional(new_dict, arg)
    #
    _update_conditional(new_dict, kwargs)
    return new_dict


def dict_merge(target_dict, src_dict, inplace=False):
    target_dict = target_dict if inplace else target_dict.copy()
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


def sorted_dict(d, sort_by_value=False):
    if sort_by_value:
        ds = {k:d[k] for k in sorted(d.values())}
    else:
        ds = {k:d[k] for k in sorted(d.keys())}
    #
    return ds


def as_tuple(arg):
    return arg if isinstance(arg, tuple) else (arg,)


def as_list(arg):
    return arg if isinstance(arg, list) else [arg]


def as_list_or_tuple(arg):
    return arg if isinstance(arg, (list,tuple)) else (arg,)


# convert to something that can be saved by yaml.safe_dump
def pretty_object(d, depth=10, precision=3):
    depth = depth - 1
    pass_through_types = (str, int)
    if depth < 0:
        d_out = None
    elif d is None:
        d_out = d
    elif isinstance(d, pass_through_types):
        d_out = d
    elif isinstance(d, (np.float32, np.float64)):
        # numpy objects cannot be serialized with yaml - convert to float
        d_out = round(float(d), precision)
    elif isinstance(d, np.int64):
        d_out = int(d)
    elif isinstance(d, float):
        # round to the given precision
        d_out = round(d, precision)
    elif isinstance(d, dict):
        d_out = {k: pretty_object(v, depth) for k , v in d.items()}
    elif isinstance(d, (list,tuple)):
        d_out = [pretty_object(di, depth) for di in d]
    elif isinstance(d, np.ndarray):
        d_out = pretty_object(d.tolist(), depth)
    elif isinstance(d, ParamsBase):
        # this is a special case
        p = d.peek_params()
        d_out = pretty_object(p, depth)
    elif hasattr(d, '__dict__'):
        # other unrecognized objects - just grab the attributes as a dict
        attrs = d.__dict__.copy()
        if 'name' not in attrs:
            attrs.update({'name':d.__class__.__name__})
        #
        d_out = pretty_object(attrs, depth)
    else:
        d_out = None
    #
    return d_out


def str_to_dict(v):
    if v is None:
        return None
    #
    if isinstance(v, list):
        v = ' '.join(v)
    #
    d = yaml.safe_load(v)
    return d


def str_to_list(v):
    if v in ('', None, 'None', 'none'):
        vs = None
    else:
        vs = v.split(' ')
    #
    return vs


def str_to_list_int(v):
    vs = str_to_list(v)
    vs = [int(i) for i in vs] if isinstance(vs, (list,tuple)) else vs
    return vs


def str_to_list_float(v):
    vs = str_to_list(v)
    vs = [float(i) for i in vs] if isinstance(vs, (list,tuple)) else vs
    return vs


def str_to_int(v):
    if v in ('', None, 'None', 'none'):
        return None
    else:
        return int(v)
    #


def str_or_bool(v):
  if isinstance(v, (str)):
      if v.lower() in ("yes", "true", "t", "1"):
          return True
      elif v.lower() in ("no", "false", "f", "0"):
          return False
      elif v.lower() in ("none",):
          return None
      else:
          return v
      #
  else:
      return v


def str_or_none(v):
    if v in (None, 'None', 'none'):
        return None
    else:
        return v


def int_or_none(v):
    if v is None or v in ('None', 'none'):
        return None
    else:
        return int(v)


def str2bool(v):
    '''a utility function used for argument parsing'''
    if v is None:
        return False
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'false', 'no', '0'):
            return False
        elif v.lower() in ('true', 'yes', '1'):
            return True
        #
    #
    return bool(v)


def str2bool_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str) and v.lower() in ('none',):
        return None
    else:
        return str2bool(v)


def splitstr2bool(v):
  v = v.split(',')
  for index, args in enumerate(v):
      v[index] = str2bool(args)
  return v


def is_url(v):
    is_url = isinstance(v, str) and (v.startswith('http://') or v.startswith('https://'))
    return is_url


def is_url_or_file(v):
    is_url_ = is_url(v)
    is_file_ = isinstance(v, str) and (v.startswith("/") or v.startswith("."))
    return is_url_ or is_file_
    

def default_arg(parser, option, value, modify_argv=True):
    '''Change the default value of action in ArgumentParser instance'''
    option1 = option
    option2 = option.replace('_', '-')
    modify_options = [option1, '-'+option1, '--'+option1, option2, '-'+option2, '--'+option2]
    for action in parser._actions:
        for option_string in action.option_strings:
            for option_name in modify_options:
                if option_string == option_name:
                    action.default = value
                    action.const = value
                    if modify_argv:
                        if isinstance(value, (list, tuple)):
                            sys.argv.insert(1, f'{option_name}')
                            for pos, v in enumerate(value):
                                sys.argv.insert(2+pos, f'{v}')
                            #
                        else:
                            sys.argv.insert(1, f'{option_name}={value}')
                        #
                    #
                    return True
                #
            #
        #
    #
    return False
