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

##############################################################################

# Also includes parts from: https://github.com/open-mmlab/mmdetection3d
# License: https://github.com/open-mmlab/mmdetection3d/blob/master/LICENSE
#
# Copyright 2018-2020 Open-MMLab. All rights reserved.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2018-2020 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

##############################################################################


import os
import random
from colorama import Fore
from .. import utils
from .dataset_base import *
import numpy as np

class KittiLidar3D(DatasetBase):
    def __init__(self, download=False, read_anno=True, dest_dir=None, num_frames=None, name='kitti_lidar_det', **kwargs):
        super().__init__(num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in self.kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'

        path = self.kwargs['path']
        split_folder = self.kwargs['split']

        from tools.data_converter import kitti_converter as kitti

        # download the data if needed
        if download:
            assert False , 'Download is not supported for this dataset'

        assert os.path.exists(path) and os.path.isdir(path), \
            utils.log_color('\nERROR', 'dataset path is empty', path)

        # create list of images and classes
        self.imgs = utils.get_data_list(input= path + "/ImageSets/val.txt", dest_dir=dest_dir)
        self.num_frames = self.kwargs['num_frames'] = self.kwargs.get('num_frames',len(self.imgs))
        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #
        if read_anno == True:
            import pickle

            if os.path.exists(os.path.join(self.kwargs['path'],'kitti_infos_val.pkl')) == False:
                self.kitti_data_prep(root_path = self.kwargs['path'],
                                info_prefix='kitti',
                                version='v1.0',
                                out_dir=self.kwargs['path'] + '/kitti_temp')

            pipeline = [{'type':'LoadPointsFromFile', 'coord_type':'LIDAR', 'load_dim':4, 'use_dim':4},
                        {'type': 'MultiScaleFlipAug3D', 'img_scale': (1333, 800), 'pts_scale_ratio': 1, 'flip': False, 'transforms': [{'type': 'GlobalRotScaleTrans', 'rot_range': [0, 0], 'scale_ratio_range': [1.0, 1.0], 'translation_std': [0, 0, 0]}, {'type': 'RandomFlip3D'}, {'type': 'PointsRangeFilter', 'point_cloud_range': [0, -39.68, -3, 69.12, 39.68, 1]}, {'type': 'DefaultFormatBundle3D', 'class_names': ['Car'], 'with_label': False}, {'type': 'Collect3D', 'keys': ['points']}]}
                        ]

            from mmdet3d.datasets import KittiDataset

            self.kitti_dataset = KittiDataset(self.kwargs['path'],os.path.join(self.kwargs['path'],'kitti_infos_val.pkl'),
                         self.kwargs['path'],
                         pipeline=pipeline,
                         pts_prefix='velodyne_reduced',
                         modality = {'use_lidar':True, 'use_camera':False},
                         classes=['Car'],
                         test_mode=True)
            assert self.num_frames <= len(self.kitti_dataset.data_infos), \
                'Number of frames is higher than length of dataset \n'

            self.kitti_dataset.data_infos = self.kitti_dataset.data_infos[:self.num_frames]

        self.num_classes = kwargs['num_classes']
        self.class_to_name = {
                                0: 'Car',
                                1: 'Pedestrian',
                                2: 'Cyclist',
                                3: 'Van',
                                4: 'Person_sitting',
                        }

    def kitti_data_prep(self, root_path, info_prefix, version, out_dir):
        """Prepare data related to Kitti dataset.

        Related data consists of '.pkl' files recording basic infos,
        2D annotations and groundtruth database.

        Args:
            root_path (str): Path of dataset root.
            info_prefix (str): The prefix of info filenames.
            version (str): Dataset version.
            out_dir (str): Output directory of the groundtruth database info.
        """
        from tools.data_converter import kitti_converter as kitti
        kitti.create_kitti_info_file(root_path, info_prefix)
        kitti.create_reduced_point_cloud(root_path, info_prefix)

        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        info_trainval_path = osp.join(root_path,
                                    f'{info_prefix}_infos_trainval.pkl')
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        kitti.export_2d_annotation(root_path, info_train_path)
        kitti.export_2d_annotation(root_path, info_val_path)
        kitti.export_2d_annotation(root_path, info_trainval_path)
        kitti.export_2d_annotation(root_path, info_test_path)

        create_groundtruth_database(
            'KittiDataset',
            root_path,
            info_prefix,
            f'{out_dir}/{info_prefix}_infos_train.pkl',
            relative_path=False,
            mask_anno_path='instances_train.json',
            with_mask=(version == 'mask'))

    def download(self, path, split_file):
        return None

    def __getitem__(self, idx, **kwargs):

        with_label = kwargs.get('with_label', False)
        words = self.imgs[idx].split(' ')
        image_name = os.path.join(self.kwargs['path'],self.kwargs['split'],self.kwargs['pts_prefix']) + '/' + words[0] + '.bin'
        if with_label:
            assert len(words)>0, f'ground truth requested, but missing at the dataset entry for {words}'
            label = int(words[1])
            return image_name, label
        else:
            return image_name
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, **kwargs):

        acc = self.kitti_dataset.evaluate(predictions, metric='mAP')
        ap_dict = {'KITTI/Car_3D_moderate_strict':acc}
        return ap_dict

    def load_kitti_3d_obj_annotation(self, anno_path, split_file):

        annos = []

        with open(split_file, 'r') as sf:
            in_files = sf.readlines()

        for file in in_files:
            annotations = {}
            annotations.update({
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': []
            })
            with open(os.path.join(anno_path, file.strip())+'.txt', 'r') as f:
                lines = f.readlines()
            # if len(lines) == 0 or len(lines[0]) < 15:
            #     content = []
            # else:
            content = [line.strip().split(' ') for line in lines]
            num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
            annotations['name'] = np.array([x[0] for x in content])
            num_gt = len(annotations['name'])
            annotations['truncated'] = np.array([float(x[1]) for x in content])
            annotations['occluded'] = np.array([int(x[2]) for x in content])
            annotations['alpha'] = np.array([float(x[3]) for x in content])
            annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                            for x in content]).reshape(-1, 4)
            # dimensions will convert hwl format to standard lhw(camera) format.
            annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                                for x in content
                                                ]).reshape(-1, 3)[:, [2, 0, 1]]
            annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                                for x in content]).reshape(-1, 3)
            annotations['rotation_y'] = np.array([float(x[14])
                                                for x in content]).reshape(-1)
            if len(content) != 0 and len(content[0]) == 16:  # have score
                annotations['score'] = np.array([float(x[15]) for x in content])
            else:
                annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)
            annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)

            annos.append(annotations)

        return annos

