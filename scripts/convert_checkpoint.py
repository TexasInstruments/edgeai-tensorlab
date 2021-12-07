import collections
import os
import collections
import re
import torch

source_checkpoint = '/data/ssd/files/a0393608/work/code/ti/edgeai-algo/edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet-lite/ssd-lite_regnetx-1.6gf_fpn_bgr_768x768_20200923-124632_checkpoint.pth'

dest_checkoint = os.path.splitext(os.path.basename(source_checkpoint))
dest_checkoint = dest_checkoint[0] + '_converted' + dest_checkoint[1]

checkpoint_dict = torch.load(source_checkpoint)
state_dict = checkpoint_dict['state_dict']


change_names_dict_ssd = {
    r'lateral_convs.(\d).0.': r'lateral_convs.\1.conv.',
    r'lateral_convs.(\d).1.': r'lateral_convs.\1.bn.',

    r'fpn_convs.(\d).0.0.': r'fpn_convs.\1.conv.0.0.',
    r'fpn_convs.(\d).0.1.': r'fpn_convs.\1.conv.0.1.',
    r'fpn_convs.(\d).1.0.': r'fpn_convs.\1.conv.1.0.',
    r'fpn_convs.(\d).1.1.': r'fpn_convs.\1.bn.',

    r'bbox_head.cls_convs.(\d).(\d).(\d).': r'bbox_head.cls_convs.\1.0.\2.\3.',
    r'bbox_head.reg_convs.(\d).(\d).(\d).': r'bbox_head.reg_convs.\1.0.\2.\3.',
}
change_names_dict = change_names_dict_ssd


state_dict_new = collections.OrderedDict()
for k, v in state_dict.items():
    for sk in change_names_dict:
        if re.search(re.compile(sk), k):
            new_k = re.sub(sk, change_names_dict[sk], k)
            break
        else:
            new_k = k
        #
    #
    print(f'{k} -> {new_k}')
    state_dict_new[new_k] = v
#

checkpoint_dict['state_dict'] = state_dict_new
torch.save(checkpoint_dict, dest_checkoint)

