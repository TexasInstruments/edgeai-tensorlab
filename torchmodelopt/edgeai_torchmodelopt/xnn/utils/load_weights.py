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

import re
import torch
import copy
from collections import OrderedDict
from . import print_utils
from . import data_utils

######################################################
# the method used in vision/models
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


######################################################
# our custom load function with more features
def load_weights(model, pretrained, change_names_dict=None, keep_original_names=False, width_mult=1.0,
                        ignore_size=True, verbose=False, num_batches_tracked = None, download_root=None,
                        state_dict_name='state_dict', **kwargs):
    device = next(model.parameters()).device

    download_root = './' if (download_root is None) else download_root
    if pretrained is None or pretrained is False:
        print_utils.print_yellow(f'=> weights could not be loaded. pretrained data given is {pretrained}')
        return model

    if isinstance(pretrained, str):
        if pretrained.startswith('http://') or pretrained.startswith('https://'):
            pretrained_file = data_utils.download_url(pretrained, root=download_root)
        else:
            pretrained_file = pretrained
        #
        data = torch.load(pretrained_file, map_location=device)
    else:
        data = pretrained
    #

    load_error = False
    state_dict_names = state_dict_name if isinstance(state_dict_name, (list,tuple)) else [state_dict_name]
    for s_name in state_dict_names:
        data = data[s_name] if ((data is not None) and s_name in data) else data

    if width_mult != 1.0:
        data = widen_model_data(data, factor=width_mult)
    #
    
    if 'state_dict' in data:
        data = data['state_dict']
    
    try:
        model.load_state_dict(data, strict=True)
    except:
        load_error = True

    if load_error:
        # model did not load correctly. do any translation required.
        model_dict = model.state_dict()

        # align the prefix 'module.' between model and data
        model_prefix = 'module.' if 'module.' in list(model_dict.keys())[0] else ''
        data_prefix = 'module.' if 'module.' in list(data.keys())[0] else ''
        data = {k.replace(data_prefix,model_prefix):v for k,v in data.items()} if data_prefix != '' \
            else {model_prefix+k:v for k,v in data.items()}

        # change the name in pretrained data name to the given names
        if change_names_dict is not None:
            new_data = copy.deepcopy(data) if keep_original_names else {}
            for data_str, model_str in change_names_dict.items():
                model_str_list = [model_str] if not isinstance(model_str,(list,tuple)) else model_str
                for new_str in model_str_list:
                    for k, v in data.items():
                        new_key = re.sub(data_str, new_str, k) if (new_str not in k) else k
                        change_names_dict_other_keys = [key for key in change_names_dict.keys() if key != data_str]
                        if new_key == k and _match_name_partial(change_names_dict_other_keys, k):
                            # this key will be dealt with later
                            pass
                        elif new_key not in new_data.keys():
                            new_data.update({new_key:v})

            data = new_data
        #

        missing_weights, extra_weights, not_matching_sizes = check_model_data(model, data, verbose=verbose)

        # num_batches_tracked was newly added to batchnorm in track_running_stats_mode (default)
        # if it is not present in the pre-trained weights, use a reasonably good value
        # so that batch norm stats won't change suddenly
        for mw in missing_weights:
            if 'num_batches_tracked' in mw:
                data[mw] = torch.tensor(100)
        #

        if ignore_size:
            try:
                model.load_state_dict(data, strict=False)
            except:
                print_utils.print_yellow('=> WARNING: weights could not be loaded completely.')
        else:
            model.load_state_dict(data, strict=False)
        #
    #
    return model


def check_model_data(model, data, verbose=False, ignore_names=('num_batches_tracked',)):
    model_dict = model.state_dict()
    missing_weights = [k for k, v in model_dict.items() if k not in list(data.keys())]
    extra_weights = [k for k, v in data.items() if k not in list(model_dict.keys())]
    
    missing_weights = [name for name in missing_weights if not _match_name_partial(ignore_names, name)]
    extra_weights = [name for name in extra_weights if not _match_name_partial(ignore_names, name)]
    not_matching_sizes = [k for k in model_dict.keys() if ((k in data.keys()) and (data[k].size() != model_dict[k].size()))]
    # for k in model_dict.keys():
    #     if ((k in data.keys()) and (data[k].size() != model_dict[k].size())):
    #         print(f"size mismatch for layer k expected:{model_dict[k].size()}, got:{data[k].size()}")

    if missing_weights:
        print_utils.print_yellow("=> The following layers in the model could not be loaded from pre-trained: ", *missing_weights, sep = "\n")
    if not_matching_sizes:
        print_utils.print_yellow("=> The shape of the following weights did not match: ", *not_matching_sizes, sep = "\n")
    if extra_weights:
        print_utils.print_yellow("=> The following weights in pre-trained were not used: ", *extra_weights, sep = "\n")
                
    return missing_weights, extra_weights, not_matching_sizes

def _match_name_partial(name_list, name):
    for name_entry in name_list:
        name_entry_pattern = re.compile(name_entry)
        if name_entry_pattern.search(name):
            return True
    #
    return False


def widen_model_data(data, factor, verbose = True):
    for k,v in data.items():
      classifier = ('fc' in k or 'classifier' in k)
      if isinstance(v, OrderedDict):
          data[k] = widen_model_data(v, factor=factor, verbose=verbose)
      elif 'shape' not in dir(v):
          continue
      elif len(v.shape)==4:
        in_xp_channels = int(v.shape[1]*factor) if v.shape[1]>3 else v.shape[1]
        xv = torch.zeros([int(v.shape[0]*factor), in_xp_channels, v.shape[2], v.shape[3]])
        for rpt in range(int(factor)):
          start = rpt*v.shape[0]
          end = (rpt+1)*v.shape[0]
          xv[start:end, :v.shape[1],...] = v
        if v.shape[1]>3:
          for rpt in range(1,int(factor)):
            start = rpt*v.shape[1]
            end = (rpt+1)*v.shape[1]
            xv[:, start:end, ...] = xv[:, :v.shape[1], ...]
          xv = xv/int(factor)
        data[k] = xv
      elif len(v.shape)==2:
        out_xp_channels = int(v.shape[0]*factor) if not classifier else v.shape[0]
        in_xp_channels = int(v.shape[1]*factor) if v.shape[1]>3 else v.shape[1]
        xv = torch.zeros([out_xp_channels, in_xp_channels])
        if not classifier:
            for rpt in range(int(factor)):
              start = rpt*v.shape[0]
              end = (rpt+1)*v.shape[0]
              xv[start:end, :v.shape[1]] = v
        if v.shape[1]>3:
          for rpt in range(1,int(factor)):
            start = rpt*v.shape[1]
            end = (rpt+1)*v.shape[1]
            xv[:, start:end, ...] = xv[:, :v.shape[1]]
          xv = xv/int(factor)
        data[k] = xv
        print(k,xv.shape)
      elif len(v.shape)==1 and not classifier:
        xv = torch.zeros([int(v.shape[0]*factor)])      
        for rpt in range(int(factor)):
          start = rpt*v.shape[0]
          end = (rpt+1)*v.shape[0]
          xv[start:end] = v
        if 'running_' not in k:
            xv = xv/int(factor)
        data[k] = xv

    return data
    
