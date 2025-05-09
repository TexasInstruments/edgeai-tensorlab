# Copyright (c) 2018-2024, Texas Instruments
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
import cv2
import tempfile

from pathlib import Path
from pyquaternion import Quaternion

import pandaset as PS

from .pandaset_object_eval_python.format_bbox import *
from .pandaset_object_eval_python.eval import *
from .pandaset_object_eval_python.utils import *

from edgeai_benchmark.postprocess.bev_detection import box3d_multiclass_nms
from edgeai_benchmark import datasets

# TODO modify these to configurable
PandaSetNameMapping = {
    'Car':'Car', 
    'Semi-truck':'Temporary Construction Barriers', 
    'Other Vehicle - Construction Vehicle':'Temporary Construction Barriers', 
    'Pedestrian with Object':'Pedestrian', 
    'Train':'Temporary Construction Barriers', 
    'Animals - Bird':'Temporary Construction Barriers', 
    'Bicycle':'Temporary Construction Barriers', 
    'Rolling Containers':'Temporary Construction Barriers', 
    'Pylons':'Temporary Construction Barriers', 
    'Signs':'Temporary Construction Barriers', 
    'Emergency Vehicle':'Temporary Construction Barriers', 
    'Towed Object':'Temporary Construction Barriers', 
    'Personal Mobility Device':'Temporary Construction Barriers', 
    'Motorcycle':'Temporary Construction Barriers', 
    'Tram / Subway':'Temporary Construction Barriers', 
    'Other Vehicle - Uncommon':'Temporary Construction Barriers', 
    'Other Vehicle - Pedicab':'Temporary Construction Barriers', 
    'Temporary Construction Barriers':'Temporary Construction Barriers', 
    'Animals - Other':'Temporary Construction Barriers', 
    'Bus':'Temporary Construction Barriers', 
    'Motorized Scooter':'Temporary Construction Barriers', 
    'Pickup Truck':'Temporary Construction Barriers', 
    'Road Barriers':'Temporary Construction Barriers', 
    'Pedestrian':'Pedestrian', 
    'Construction Signs':'Temporary Construction Barriers', 
    'Cones':'Temporary Construction Barriers', 
    'Medium-sized Truck':'Temporary Construction Barriers'
}


ErrNameMapping = {
    'trans_err': 'mATE',
    'scale_err': 'mASE',
    'orient_err': 'mAOE',
    'vel_err': 'mAVE',
    'attr_err': 'mAAE'
}


# FCOS3D, FastBEV
class_to_name_type= {
    0: 'Car',
    1: 'Pedestrian',
    2: 'Bus', 
    # 3: 'Pedestrian',
    # 4: 'Temporary Construction Barriers'
}


class PandasetDatasetBase:
    AVAILABLE_VERSIONS: dict[str, list[str]] ={
        'v1.0-full':None,
        'v1.0-mini':['001','003'],
    }
    def __init__(self, path, version):
        assert version in self.AVAILABLE_VERSIONS, \
            utils.log_color('\nERROR', 'dataset version is not supported', version)
        self.dataset = PS.DataSet(path)
        self.version = version
        if version == 'v1.0-full':
            self.scene_list = self.dataset.sequences()
        else:
            self.scene_list = self.AVAILABLE_VERSIONS[version]
    
    def __getitem__(self, seq_id):
        if seq_id not in self.scene_list:
            raise KeyError(f"Sequence {seq_id} not found in dataset.")
        return self.dataset[seq_id]

    def __len__(self):
        return len(self.scene_list)


def load_pandaset(path):
    assert os.path.exists(path) and os.path.isdir(path), \
        utils.log_color('\nERROR', 'dataset path is empty, and cannot load pandaset dataset', path)

    dataset = PandasetDatasetBase(path,'v1.0-mini')
    return dataset


def get_frame_token(seq_id:str, frame_id:int):
    return seq_id + f'_{frame_id:02}'


def get_available_scenes(ps:PandasetDatasetBase):
    seqs = ps.scene_list
    print('total scene num: {}'.format(len(seqs)))
    available_scenes = []
    for s_id in seqs:
        seq = ps[s_id]
        available_scenes.append(dict(
            token=s_id,
            nbr_samples=len(seq.lidar._data_structure),
            name=s_id,
            first_sample_token=get_frame_token(s_id, 0),
            last_sample_token=get_frame_token(s_id, len(seq.lidar._data_structure)-1),
            description=f'scene_details of scene {s_id}'
        ))
        
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(ps: PandasetDatasetBase,
                            train_scenes=None,
                            val_scenes=None,
                            data_ids=None,
                            read_anno=True,):
    ps_infos = []
    for scene in ps.scene_list:
        if train_scenes is not None and scene not in train_scenes:
            continue
        if val_scenes is not None and scene not in val_scenes:
            continue
        seq = ps[scene]
        seq.load()
        all_instances = get_gt_for_seq(seq)
        cam_intrinsics = {}
        for name, camera in seq.camera.items():
            intrinsics = camera.intrinsics
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = intrinsics.fx
            K[1, 1] = intrinsics.fy
            K[0, 2] = intrinsics.cx
            K[1, 2] = intrinsics.cy
            cam_intrinsics[name] = K
        
        for frame_idx in range(len(seq.lidar._data_structure)):
            timestamp = seq.lidar.timestamps[frame_idx]
            lidar_token = scene+f'_{frame_idx:02}'

            # lidar = ego = sensor
            lidar2global = PS.geometry._heading_position_to_mat(**seq.lidar.poses[frame_idx])
            global2lidar = np.linalg.inv(lidar2global)
            front_cam2global = PS.geometry._heading_position_to_mat(**seq.camera['front_camera'].poses[frame_idx])
            
            lidar_path=seq.lidar._data_structure[frame_idx]
            lidar_path = lidar_path if not lidar_path.endswith('.gz') else lidar_path[:-3]
            
            if not (os.path.exists(lidar_path) and os.path.isfile(lidar_path)):
                raise FileNotFoundError('file "{}" does not exist, please download PandaSet Dataset and run "run_pandaset_unzip.sh"'.format(lidar_path))
            
            
            if not os.path.isfile(lidar_path):
                raise FileNotFoundError('file "{}" does not exist'.format(lidar_path))
            info = dict(lidar_path=lidar_path,
                        num_features=4,
                        token=lidar_token,
                        timestamp=timestamp,
                        sweeps=[],
                        cams={},
                        can_bus=np.zeros(18, dtype=float),
                        lidar2ego_translation=[0,0,0],
                        lidar2ego_rotation=Quaternion(matrix=np.eye(3)).q.tolist(),
                        ego2global_translation=lidar2global[:3, 3].tolist(),
                        ego2global_rotation=Quaternion(matrix=lidar2global[:3, :3]).q.tolist(),
                        prev=scene+f'_{frame_idx-1:02}' if frame_idx>0 else None,
                        next=scene+f'_{frame_idx+1:02}' if frame_idx<len(seq.lidar._data_structure)-1 else None,
                        scene_token=scene,
                        data={name:f'{scene}_{name}_{frame_idx:02}' for name in (list(seq.camera.keys())+['lidar'])},
                        ego2global=lidar2global.tolist(),
                        lidar2global=lidar2global.tolist(),
                        lidar2ego=np.eye(4).tolist(),
                        )
            info['cams'] = {}
            for name, camera in seq.camera.items():
                cam2global = PS.geometry._heading_position_to_mat(**camera.poses[frame_idx])
                global2cam = np.linalg.inv(cam2global)
                img_path = camera._data_structure[frame_idx]
                cam_token = f'{scene}_{name}_{frame_idx:02}'
                cam2lidar = global2lidar @ cam2global
                lidar2cam = np.linalg.inv(cam2lidar)
                # sensor = cam
                cam_info = dict(
                    data_path=img_path,
                    type=name,
                    sample_data_token=cam_token,
                    sensor2ego_translation=cam2lidar[:3, 3].tolist(),
                    sensor2ego_rotation=Quaternion(matrix=cam2lidar[:3, :3]).q.tolist(),
                    ego2global_translation=lidar2global[:3, 3].tolist(),
                    ego2global_rotation=Quaternion(matrix=lidar2global[:3, :3]).q.tolist(),
                    timestamp=camera.timestamps[frame_idx],
                    sensor2lidar_translation=cam2lidar[:3, 3],
                    sensor2lidar_rotation=Quaternion(matrix=cam2lidar[:3, :3]).q,
                    lidar2sensor=lidar2cam.tolist(),
                    cam2global=cam2global.tolist(),
                    cam2ego=cam2lidar.tolist(),
                    cam_intrinsic=cam_intrinsics[name],
                    front_cam2global=front_cam2global.tolist(),
                )
                if read_anno:
                    info['anns']=all_instances[frame_idx]
                    for instance in info['anns']:
                        instance['bbox_label']=PandaSetNameMapping.get(instance['bbox_label'],'Temporary Construction Barriers')
                info['cams'][name] = (cam_info)
            ps_infos.append(info)
        
        for frame_idx in range(len(seq.lidar._data_structure)):
            del seq.cuboids.data[0]
            del seq.lidar.data[0]
            del seq.lidar.poses[0]
            for camera in seq.camera.values():
                del camera.data[0]
                del camera.poses[0]
            if seq.semseg:
                del seq.semseg.data[0]
    return ps_infos


def _fill_trainval_infos_mv_image(ps: PandasetDatasetBase,
                            train_scenes=None,
                            val_scenes=None,
                            data_ids=None,
                            read_anno=True,):
    ps_infos = []
    for scene in ps.scene_list:
        if train_scenes is not None and scene not in train_scenes:
            continue
        if val_scenes is not None and scene not in val_scenes:
            continue
        seq = ps[scene]
        seq.load()
        all_instances = get_gt_for_seq(seq)
        
        cam_intrinsics = {}
        for name, camera in seq.camera.items():
            intrinsics = camera.intrinsics
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = intrinsics.fx
            K[1, 1] = intrinsics.fy
            K[0, 2] = intrinsics.cx
            K[1, 2] = intrinsics.cy
            cam_intrinsics[name] = K
        
        
        for frame_idx in range(len(seq.lidar._data_structure)):
            # lidar = ego
            token = f'{scene}_{frame_idx:02}'
            lidar2global = PS.geometry._heading_position_to_mat(**seq.lidar.poses[frame_idx])
            global2lidar = np.linalg.inv(lidar2global)
            front_cam2global = PS.geometry._heading_position_to_mat(**seq.camera['front_camera'].poses[frame_idx])
            for name, camera in seq.camera.items():
                cam2global = PS.geometry._heading_position_to_mat(**camera.poses[frame_idx])
                global2cam = np.linalg.inv(cam2global)
                img_path = camera._data_structure[frame_idx]
                cam_token = f'{scene}_{name}_{frame_idx:02}'
                cam2lidar = global2lidar @ cam2global
                lidar2cam = np.linalg.inv(cam2lidar)
                # sensor = cam
                cam_info = dict(
                    data_path=img_path,
                    type=name,
                    sample_data_token=cam_token,
                    sensor2ego_translation=cam2lidar[:3, 3].tolist(),
                    sensor2ego_rotation=Quaternion(matrix=cam2lidar[:3, :3]).q.tolist(),
                    ego2global_translation=lidar2global[:3, 3].tolist(),
                    ego2global_rotation=Quaternion(matrix=lidar2global[:3, :3]).q.tolist(),
                    timestamp=camera.timestamps[frame_idx],
                    sensor2lidar_translation=cam2lidar[:3, 3],
                    sensor2lidar_rotation=Quaternion(matrix=cam2lidar[:3, :3]).q,
                    lidar2sensor=lidar2cam.tolist(),
                    cam2ego=cam2lidar.tolist(),
                    cam2global=cam2global.tolist(),
                    cam_intrinsic=cam_intrinsics[name],
                    front_cam2global=front_cam2global.tolist(),
                )
                camera_info = dict(
                    images={name:cam_info}, 
                    camera_type=name, 
                    ego2global=lidar2global.tolist(), 
                    timestamp=camera.timestamps[frame_idx],
                    token=token
                    )
                if read_anno:
                    camera_info['anns']=all_instances[frame_idx]
                    for instance in camera_info['anns']:
                        instance['bbox_label']=PandaSetNameMapping.get(instance['bbox_label'],'Temporary Construction Barriers')
                ps_infos.append(camera_info)
        
        for frame_idx in range(len(seq.lidar._data_structure)):
            del seq.cuboids.data[0]
            del seq.lidar.data[0]
            del seq.lidar.poses[0]
            for camera in seq.camera.values():
                del camera.data[0]
                del camera.poses[0]
            if seq.semseg:
                del seq.semseg.data[0]
    return ps_infos


class PandaSetDataset(DatasetBase):
    def __init__(self, ps=None,
                 download=False, read_anno=True, dest_dir=None, num_frames=None,  **kwargs):
        super().__init__(**kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        
        path = self.kwargs['path']
        split_folder = self.kwargs['split']
        # load_type: frame_based, mv_image_based, fov_image_based
        self.load_type = self.kwargs['load_type']

        # download the data if needed
        if download:
            assert False , 'Download is not supported for this dataset'

        #assert os.path.exists(path) and os.path.isdir(path), \
        #    utils.log_color('\nERROR', 'dataset path is empty', path)

        # pandaset dataset
        assert ps is not None, '\nERROR, nucenes dataset is None'
        self.ps = ps

        # create list of images and classes
        self.data_ids = list(np.arange(0, 404))

        self.num_frames = self.kwargs['num_frames'] = self.kwargs.get('num_frames',len(self.data_ids))

        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.data_ids)

        self.num_classes = kwargs['num_classes']
        
        self.data_infos, self.data_scene_infos = self.create_pandaset_infos(read_anno, split_folder)
        
        # For validation dataset, read annotaiton for evaluation
        # TODO modify this to configurable
        if read_anno:
            self.eval_version='detection_cvpr_2019'
            self.eval_detection_configs = dict(
                class_names=list(class_to_name_type.values()),
                class_range={c:50 for c in class_to_name_type.values()},
                dist_fcn='center_distance',
                dist_th_tp=2.0,
                dist_ths=[0.5,1,2,4],
                max_boxes_per_sample=500,
                mean_ap_weight=5,
                min_precision=0.1,
                min_recall=0.1,
                )

            self.DefaultAttribute = {
                'Car':'vehicle.Parked',
                # 'Bus':'vehicle.Parked',
                # 'Bicycle':'vehicle.Parked',
                'Pedestrian':'pedestrian.Standing',
                'Temporary Construction Barriers':'None'
            }
            self.metrics=['bbox']
        
        self.max_dists = 50

    def download(self, path, split_file):
        return None

    def __getitem__(self, idx, **kwargs):
        return self.data_infos['infos'][idx]

    def __len__(self):
        return len(self.data_infos['infos'])
    
    def create_pandaset_infos(self, read_anno=True, split_folder='train', version='v1.0-mini', max_sweeps=10):
        metadata = dict(version=version)
        if split_folder == 'train':
            train_scenes = ['001']
            available_scenes = get_available_scenes(self.ps)
            available_scene_names = [s['name'] for s in available_scenes]
            # filter existing scenes.
            train_scenes = list(
                filter(lambda x: x in available_scene_names, train_scenes))

            train_scenes = set([
                available_scenes[available_scene_names.index(s)]['token']
                for s in train_scenes
            ])

            if self.load_type == 'mv_image_based':
                train_ps_infos = _fill_trainval_infos_mv_image(self.ps, train_scenes=train_scenes, val_scenes=None,
                    data_ids=self.data_ids, read_anno=read_anno)
            else:
                train_ps_infos = _fill_trainval_infos(self.ps, train_scenes=train_scenes, val_scenes=None,
                    data_ids=self.data_ids, read_anno=read_anno,)
                # Sort with timestamp
                train_ps_infos = list(sorted(train_ps_infos, key=lambda e: e['timestamp']))

            print('train sample: {}'.format(len(train_ps_infos)))
            data = dict(infos=train_ps_infos, metadata=metadata)

            return data, train_scenes
        else:
            val_scenes = ['003']
            available_scenes = get_available_scenes(self.ps)
            available_scene_names = [s['name'] for s in available_scenes]
            # filter existing scenes.
            val_scenes = list(
                filter(lambda x: x in available_scene_names, val_scenes))

            val_scenes = set([
                available_scenes[available_scene_names.index(s)]['token']
                for s in val_scenes
            ])

            if self.load_type == 'mv_image_based':
                val_ps_infos = _fill_trainval_infos_mv_image(self.ps, train_scenes=None, val_scenes=val_scenes,
                    data_ids=self.data_ids, read_anno=read_anno)
            else:
                val_ps_infos = _fill_trainval_infos(self.ps,  train_scenes=None, val_scenes=val_scenes,
                    data_ids=self.data_ids, read_anno=read_anno,)
                # Sort with timestamp
                val_ps_infos = list(sorted(val_ps_infos, key=lambda e: e['timestamp']))

            print('val sample: {}'.format(len(val_ps_infos)))
            data = dict(infos=val_ps_infos, metadata=metadata)

            return data, val_scenes
    
    def format_results_lidar_bbox(self, predictions, det_classes, **kwargs):
        pandaset_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(predictions):
            annos = []
            boxes = det['bboxes_3d']
            # make (0.5, 0.5, 0.5) center
            boxes[:, 2] += boxes[:, 5] * 0.5
            if 'attr_labels' in det:
                attrs = det['attr_labels'].tolist()
            else:
                attrs = None
            scores = det['scores_3d'].tolist()
            labels = det['labels_3d'].tolist()
            sample_idx = i
            info = self.data_infos['infos'][sample_idx]
            sample_token = info['token']
            boxes = convert_lidar_box_to_global_box(boxes, info)

            for i in range(boxes.shape[0]):
                box = boxes[i]
                name = det_classes[labels[i]]
                if attrs:
                    attr = self.get_attr_name(attrs[i], name)
                elif np.linalg.norm(box[7:9]) > 0.2:
                    if name in [
                            'Car',
                            'Pickup Truck'
                            'Medium-sized Truck',
                            'Semi-truck',
                            'Towed Object',
                            'Other Vehicle - Construction Vehicle',
                            'Other Vehicle - Uncommon',
                            'Other Vehicle - Pedicab',
                            'Bus',
                            'Train',
                            'Trolley',
                            'Tram / Subway',
                            'Motorcycle',
                            'Personal Mobility Device',
                            'Motorized Scooter',
                            'Bicycle',
                            'Animals - Other',
                    ]:
                        attr = 'vehicle.Moving'
                    elif name in ['Emergency Vehicle']:
                        attr = 'emergency_vehicle.Moving'
                    elif name in [
                            'Pedestrian',
                            'Pedestrian with Object'
                    ]:
                        attr = 'pedestrian.Walking'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    attr = self.DefaultAttribute[name]
                
                pandaset_anno = dict(
                    sample_token=sample_token,
                    translation=box[0:3].tolist(),
                    size=box[3:6].tolist(),
                    yaw=box[6].tolist(),
                    velocity=box[7:9].tolist(),
                    detection_name=name,
                    detection_score=scores[i],
                    attribute_name=attr)
                annos.append(pandaset_anno)
            pandaset_annos[sample_token] = annos
        return pandaset_annos
    def format_results_camera_bbox(self, predictions, det_classes, **kwargs):
        
        pandaset_annos = {}

        print('\nStart to convert detection format...')

        # Camera types in pandaset datasets
        camera_types = [
            'back_camera',
            'left_camera',
            'front_right_camera',
            'front_left_camera',
            'right_camera',
            'front_camera',
        ]
        CAM_NUM = 6
        NUM_CLASSES = len(det_classes)

        for i, det in enumerate(predictions):
            sample_idx = i
            camera_type_id = sample_idx % CAM_NUM
            camera_type = camera_types[camera_type_id]

            if camera_type_id == 0:
                corners_per_frame = []
                velocities_per_frame = []
                attrs_per_frame = []
                labels_per_frame = []
                scores_per_frame = []

            # need to merge results from images of the same sample
            info = self.data_infos['infos'][i]
            sample_token = info['token']
            annos = []
            boxes = det['bboxes_3d']
            # boxes.tensor[:,1] = boxes.tensor[:,1] - boxes.tensor[:,4] * 0.5
            global_corners, velocities = cam_bbox_to_global_corners3d(boxes, info, camera_type)
            attrs = det['attr_labels'].tolist()
            scores = det['scores_3d'].tolist()
            labels = det['labels_3d'].tolist()
            corners_per_frame.extend(global_corners.tolist())
            velocities_per_frame.extend(velocities.tolist())
            attrs_per_frame.extend(attrs)
            labels_per_frame.extend(labels)
            scores_per_frame.extend(scores)
            # Remove redundant predictions caused by overlap of images
            if (sample_idx + 1) % CAM_NUM != 0:
                continue
            assert len(corners_per_frame) == len(velocities_per_frame) == len(attrs_per_frame) == len(labels_per_frame) == len(scores_per_frame)
            # cam_boxes3d = get_cam3d_boxes_per_frame(boxes_per_frame)
            scores= np.array(scores_per_frame)
            nms_scores = np.zeros([scores.shape[0],NUM_CLASSES+1])
            labels = np.array(labels_per_frame)
            indices = np.array(list(range(scores.shape[0])))
            nms_scores[indices,labels] = scores
            scores = nms_scores
            global_corners= np.array(corners_per_frame)
            velocities = velocities_per_frame
            cam_boxes3d = global_corners3d_to_cam_bbox(global_corners, velocities, info)
            
            # box nms 3d over 6 images in a frame
            # TODO: move this global setting into config
            nms_cfg = dict(
                use_rotate_nms=False,
                nms_across_levels=False,
                nms_pre=4096,
                nms_thr=0.05,
                score_thr=0.01,
                min_bbox_size=0,
                max_per_frame=500)
            bev = np.copy(cam_boxes3d[:, [0, 2, 3, 5, 6]])
            cam_boxes3d_for_nms = xywhr2xyxyr(bev)
            # boxes3d = cam_boxes3d.tensor
            # generate attr scores from attr labels
            attrs = np.array([attr for attr in attrs_per_frame])
            boxes3d, scores, labels, attrs = box3d_multiclass_nms(
                cam_boxes3d,
                cam_boxes3d_for_nms,
                scores,
                nms_cfg['score_thr'],
                nms_cfg['max_per_frame'],
                nms_cfg,
                mlvl_attr_scores=attrs)
            det = bbox3d2result(boxes3d, scores, labels, attrs)
            
            boxes = det['bboxes_3d']
            attrs = det['attr_labels'].tolist()
            scores = det['scores_3d'].tolist()
            labels = det['labels_3d'].tolist()
            global_corners , velocities = front_cam_bbox_to_global_corners3d(boxes, info)
            boxes = global_corners3d_to_global_bbox(global_corners, velocities, )
            
            for i in range(boxes.shape[0]):
                box = boxes[i]

                name = det_classes[labels[i]]
                attr = self.get_attr_name(attrs[i], name)
                ps_anno = dict(
                    sample_token=sample_token,
                    translation=box[0:3].tolist(),
                    size=box[3:6].tolist(),
                    yaw=box[6].tolist(),
                    velocity=box[7:9].tolist(),
                    detection_name=name,
                    detection_score=scores[i],
                    attribute_name=attr)
                annos.append(ps_anno)
            # other views results of the same frame should be concatenated
            if sample_token in pandaset_annos:
                pandaset_annos[sample_token].extend(annos)
            else:
                pandaset_annos[sample_token] = annos

        return pandaset_annos
    
    def __call__(self, predictions, **kwargs):
        result_dict = dict()

        class_to_name = class_to_name_type
        det_classes = [class_to_name[i] for i in range(self.num_classes)]

        if kwargs['dataset_category'] == datasets.DATASET_CATEGORY_PANDASET_FRAME:
            result_dict['pred_instances_3d'] = \
                self.format_results_lidar_bbox(predictions, det_classes, **kwargs)
        else:
            result_dict['pred_instances_3d'] = \
                self.format_results_camera_bbox(predictions, det_classes, **kwargs)


        metric_dict = {}
        for metric in self.metrics:
            ap_dict = self.ps_evaluate(
                result_dict, det_classes=det_classes, metric=metric)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        return metric_dict
    
    def ps_evaluate(self, 
                     result_dict,
                     det_classes = None,
                     metric = 'bbox'):
        metric_dict = dict()
        for name in result_dict:
            print(f'Evaluating bboxes of {name}')
            ret_dict = self._evaluate_single(
                result_dict[name], det_classes=det_classes, result_name=name)
            metric_dict.update(ret_dict)

        return metric_dict
    
    def _evaluate_single(self,
                         pred_boxes,
                         det_classes = None,
                         result_name = 'pred_instances_3d'):
        
        metrics = self.pandaset_evaluate(pred_boxes,det_classes)

        detail = dict()
        metric_prefix = f'{result_name}'
        for name in det_classes:
            for k, v in metrics['mean_dist_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']

        return detail

    def pandaset_evaluate(self, pred_boxes, classes):
        gt_boxes = self.load_gt_bboxes()
        if self.max_dists is None:
            self.max_dist_func = lambda cls: float('inf')
        elif isinstance(self.max_dists, dict):
            self.max_dists = {((classes[key]) if isinstance(key, int)else key):val for key, val in self.max_dists.items()}
            self.max_dist_func = lambda cls: self.max_dists.get(cls, 50)
        elif isinstance (self.max_dists, (int, float)):
            self.max_dist_func = lambda cls: self.max_dists
        elif isinstance(self.max_dists, (tuple, list)):
            self.max_dists = dict(zip(classes,self.max_dists))
            self.max_dist_func = lambda cls: self.max_dists.get(cls, 50)
            
        gt_boxes = filter_eval_boxes(gt_boxes, self.max_dist_func)
        pred_boxes = filter_eval_boxes(pred_boxes, self.max_dist_func)
        dist_ths = [0.5, 1.0, 2.0, 4.0]
        metrics = pandaset_evaluate_metrics(pred_boxes, gt_boxes, classes, dist_ths, 2.0)
        return metrics
    
    def get_attr_name(self, attr_idx, label_name):
        attr_mapping = UNIQUE_ATTRIBUTE_LABELS
        if attr_idx == -1:
            return 'None'
        
        if label_name in ('Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck',
                'Towed Object', 'Other Vehicle - Construction Vehicle',
                'Other Vehicle - Uncommon', 'Other Vehicle - Pedicab',
                'Bus', 'Train', 'Trolley', 'Tram / Subway',):
            if attr_idx in list(range(9, 18)):
                return attr_mapping[attr_idx]
            return self.DefaultAttribute[label_name]
        
        if label_name == 'Emergency Vehicle':
            if attr_idx in list(range(5,9)):
                return attr_mapping[attr_idx]
            return self.DefaultAttribute[label_name]
        
        if label_name in ('Pedestrian', 'Pedestrian with Object'):
            if attr_idx in list(range(4)):
                return attr_mapping[attr_idx]
            return self.DefaultAttribute[label_name]
        
        if label_name in ('Motorcycle', 'Personal Mobility Device', 'Motorized Scooter',
                        'Bicycle', 'Animals - Other',):
            if attr_idx in list(range(9,18)):
                return attr_mapping[attr_idx]
            return self.DefaultAttribute[label_name]
        
        return 'None'
    
    def load_gt_bboxes(self):
        infos = self.data_infos['infos']
        gt_bboxes = {}
        for info in infos:
            token = info['token']
            anns = info['anns']
            annos = []
            for ann in anns:
                annos.append(dict(
                    sample_token=ann['token'],
                    translation=ann['bbox3d'][0:3],
                    size=ann['bbox3d'][3:6],
                    yaw=ann['bbox3d'][6],
                    velocity=ann['velocity'][:2],
                    detection_name=ann['bbox_label'],
                    attribute_name=self.get_attr_name(ann['attr_label'],ann['bbox_label']),
                ))
            gt_bboxes[token] = annos
        return gt_bboxes