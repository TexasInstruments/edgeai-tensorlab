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

from nuscenes.eval.detection.config import config_factory

from .nuscenes_object_eval_python.format_bbox import *
from .nuscenes_object_eval_python.utils import *
from .nuscenes_object_eval_python.eval import *

from edgeai_benchmark.postprocess.bev_detection import box3d_multiclass_nms
from edgeai_benchmark import datasets

NuScenesNameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

ErrNameMapping = {
    'trans_err': 'mATE',
    'scale_err': 'mASE',
    'orient_err': 'mAOE',
    'vel_err': 'mAVE',
    'attr_err': 'mAAE'
}

# FCOS3D, FastBEV
class_to_name_type_1= {
    0: 'car',
    1: 'truck',
    2: 'trailer',
    3: 'bus',
    4: 'construction_vehicle',
    5: 'bicycle',
    6: 'motorcycle',
    7: 'pedestrian',
    8: 'traffic_cone',
    9: 'barrier'
}

# Other BEV networks
class_to_name_type_2 = {
    0: 'car',
    1: 'truck',
    2: 'construction_vehicle',
    3: 'bus',
    4: 'trailer',
    5: 'barrier',
    6: 'motorcycle',
    7: 'bicycle',
    8: 'pedestrian',
    9: 'traffic_cone'
}


def load_nuscenes(path):
    assert os.path.exists(path) and os.path.isdir(path), \
        utils.log_color('\nERROR', 'dataset path is empty, and cannot load nuscenes dataset', path)

    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    nusc = NuScenes(version='v1.0-mini', dataroot=os.path.join(path, 'nuscenes'), verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=path)

    return nusc, nusc_can_bus



# https://github.com/open-mmlab/mmdetection3d/
# From get_available_scenes()
def get_available_scenes(nusc: NuScenes):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not (isinstance(lidar_path, str) or isinstance(lidar_path, Path)):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


# https://github.com/open-mmlab/mmdetection3d/
# From _fill_trainval_infos()
def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes=None,
                         val_scenes=None,
                         data_ids=None,
                         read_anno=True,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    nusc_infos = []

    for idx, sample in enumerate(nusc.sample):
        if not (str(idx) in data_ids or idx in data_ids):
            continue

        if train_scenes is not None and sample['scene_token'] not in train_scenes:
            continue
        elif val_scenes is not None and sample['scene_token'] not in val_scenes:
            continue

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        if not os.path.isfile(lidar_path):
            raise FileNotFoundError('file "{}" does not exist'.format(lidar_path))

        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'can_bus': can_bus,
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'prev': sample['prev'],
            'next': sample['next'],
            'scene_token': sample['scene_token'],
            'data': sample['data'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        info['ego2global'] = convert_quaternion_to_matrix(
            info['ego2global_rotation'], info['ego2global_translation'])
        info['lidar2ego'] = convert_quaternion_to_matrix(
            info['lidar2ego_rotation'], info['lidar2ego_translation'])


        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # obtain annotation
        if read_anno:
            info['anns'] = sample['anns']

        nusc_infos.append(info)

    return nusc_infos


# https://github.com/open-mmlab/mmdetection3d/
# From _fill_trainval_infos()
def _fill_trainval_infos_mv_image(nusc,
                                  train_scenes=None,
                                  val_scenes=None,
                                  data_ids=None,
                                  read_anno=True):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    nusc_infos = []

    for idx, sample in enumerate(nusc.sample):
        if not (str(idx) in data_ids or idx in data_ids):
            continue

        if train_scenes is not None and sample['scene_token'] not in train_scenes:
            continue
        elif val_scenes is not None and sample['scene_token'] not in val_scenes: 
            continue

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        if not os.path.isfile(lidar_path):
            raise FileNotFoundError('file "{}" does not exist'.format(lidar_path))

        info = {
            'token': sample['token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'scene_token': sample['scene_token'],
            'data': sample['data'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        info['ego2global'] = convert_quaternion_to_matrix(
            info['ego2global_rotation'], info['ego2global_translation'])
        info['lidar2ego'] = convert_quaternion_to_matrix(
            info['lidar2ego_rotation'], info['lidar2ego_translation'])

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)

            if cam == 'CAM_FRONT':
                front_cam2ego = cam_info['cam2ego']
            cam_info.update(front_cam2ego=front_cam2ego)

            camera_info = dict()
            camera_info['images'] = dict()
            camera_info['images'][cam] = cam_info
            camera_info['camera_type'] = cam
            camera_info['ego2global'] = info['ego2global']
            camera_info['timestamp'] = camera_info['images'][cam]['timestamp']
            camera_info['token'] = info['token']

             # obtain annotation
             # it is redundant since we keep the same 'anns' for 6 cameras
            if read_anno:
                camera_info['anns'] = sample['anns']

            nusc_infos.append(camera_info)

    return nusc_infos


# https://github.com/open-mmlab/mmdetection3d/
# From obtain_sensor2top()
def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T

    # lidar2sensor
    lidar2sensor = np.eye(4)
    lidar2sensor[:3, :3] = R
    lidar2sensor[:3, 3:4] = -1 * np.matmul(R, T.reshape(3, 1))
    sweep['lidar2sensor'] = lidar2sensor.astype(np.float32)

    # cam2ego (sensor2ego)
    sweep['cam2ego'] = convert_quaternion_to_matrix(l2e_r_s, l2e_t_s)

    return sweep


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.

    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


class NuScenesDataset(DatasetBase):
    def __init__(self, nusc=None, nusc_can_bus=None,
                 download=False, read_anno=True, dest_dir=None, num_frames=None, name='nuscnes', **kwargs):
        super().__init__(num_frames=num_frames, name=name, read_anno=read_anno, **kwargs)

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

        # nuscenes dataset
        assert nusc is not None, '\nERROR, nucenes dataset is None'
        self.nusc = nusc

        # nuscenes dataset
        #assert nusc_can_bus is not None, '\nERROR, nucenes_can_bus is None'
        self.nusc_can_bus = nusc_can_bus

        # create list of images and classes
        if os.path.exists(path + "/data_ids.txt"):
            self.data_ids = utils.get_data_list(input= path + "/data_ids.txt", dest_dir=dest_dir)
        else:
            self.data_ids = list(np.arange(0, 404))

        self.num_frames = self.kwargs['num_frames'] = self.kwargs.get('num_frames',len(self.data_ids))

        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.data_ids)

        self.num_classes = kwargs['num_classes']
        # FCOS3D, FastBEV
        #self.class_to_name = {
        #    0: 'car',
        #    1: 'truck',
        #    2: 'trailer',
        #    3: 'bus',
        #    4: 'construction_vehicle',
        #    5: 'bicycle',
        #    6: 'motorcycle',
        #    7: 'pedestrian',
        #    8: 'traffic_cone',
        #    9: 'barrier'
        #}
        #self.classes = [self.class_to_name[i] for i in range(self.num_classes)]
        self.data_infos, self.data_scene_infos = self.create_nuscenes_infos(read_anno, split_folder)

        # For validation dataset, read annotaiton for evaluation
        if read_anno:
            self.eval_version='detection_cvpr_2019'
            self.eval_detection_configs = config_factory(self.eval_version)
            # Cast to list from dict_keys since
            # it caused error in deepcopying dataset in pipeline_runner.py, e.g. in v10.1
            # pipeline_config['input_dataset'] = copy.deepcopy(self.settings.dataset_cache[input_dataset_category]['input_dataset'])
            self.eval_detection_configs.class_names = list(self.eval_detection_configs.class_names)

            self.DefaultAttribute = {
                'car': 'vehicle.parked',
                'pedestrian': 'pedestrian.moving',
                'trailer': 'vehicle.parked',
                'truck': 'vehicle.parked',
                'bus': 'vehicle.moving',
                'motorcycle': 'cycle.without_rider',
                'construction_vehicle': 'vehicle.parked',
                'bicycle': 'cycle.without_rider',
                'barrier': '',
                'traffic_cone': '',
            }
            self.metrics=['bbox']

        #assert self.num_frames <= len(self.data_infos['infos']), \
        #    'Number of frames is higher than length of data avialable \n'


    def download(self, path, split_file):
        return None

    def __getitem__(self, idx, **kwargs):
        return self.data_infos['infos'][idx]

    def __len__(self):
        return len(self.data_infos['infos'])

    # TO BE UPDATED for NuSCenes

    # https://github.com/open-mmlab/mmdetection3d/
    # Based on create_nuscenes_infos()
    def create_nuscenes_infos(self, read_anno=True, split_folder='train', version='v1.0-mini', max_sweeps=10):

        #from nuscenes.nuscenes import NuScenes
        #nusc = NuScenes(version='v1.0-mini', dataroot=os.path.join(self.kwargs['path'], 'nuscenes'), verbose=True)
        from nuscenes.utils import splits

        metadata = dict(version=version)

        if split_folder == 'train':
            train_scenes = splits.mini_train
            available_scenes = get_available_scenes(self.nusc)
            available_scene_names = [s['name'] for s in available_scenes]
            # filter existing scenes.
            train_scenes = list(
                filter(lambda x: x in available_scene_names, train_scenes))

            train_scenes = set([
                available_scenes[available_scene_names.index(s)]['token']
                for s in train_scenes
            ])

            if self.load_type == 'mv_image_based':
                train_nusc_infos = _fill_trainval_infos_mv_image(self.nusc, train_scenes=train_scenes, val_scenes=None,
                    data_ids=self.data_ids, read_anno=read_anno)
            else:
                train_nusc_infos = _fill_trainval_infos(self.nusc, self.nusc_can_bus, train_scenes=train_scenes, val_scenes=None,
                    data_ids=self.data_ids, read_anno=read_anno, max_sweeps=max_sweeps)
                # Sort with timestamp
                train_nusc_infos = list(sorted(train_nusc_infos, key=lambda e: e['timestamp']))

            print('train sample: {}'.format(len(train_nusc_infos)))
            data = dict(infos=train_nusc_infos, metadata=metadata)

            return data, train_scenes
        else:
            val_scenes = splits.mini_val
            available_scenes = get_available_scenes(self.nusc)
            available_scene_names = [s['name'] for s in available_scenes]
            # filter existing scenes.
            val_scenes = list(
                filter(lambda x: x in available_scene_names, val_scenes))

            val_scenes = set([
                available_scenes[available_scene_names.index(s)]['token']
                for s in val_scenes
            ])

            if self.load_type == 'mv_image_based':
                val_nusc_infos = _fill_trainval_infos_mv_image(self.nusc, train_scenes=None, val_scenes=val_scenes,
                    data_ids=self.data_ids, read_anno=read_anno)
            else:
                val_nusc_infos = _fill_trainval_infos(self.nusc, self.nusc_can_bus, train_scenes=None, val_scenes=val_scenes,
                    data_ids=self.data_ids, read_anno=read_anno, max_sweeps=max_sweeps)
                # Sort with timestamp
                val_nusc_infos = list(sorted(val_nusc_infos, key=lambda e: e['timestamp']))

            print('val sample: {}'.format(len(val_nusc_infos)))
            data = dict(infos=val_nusc_infos, metadata=metadata)

            return data, val_scenes


    def __call__(self, predictions, **kwargs):
        result_dict = dict()

        if kwargs['task_name'] == 'FCOS3D' or \
            kwargs['task_name'] == 'FastBEV_f1' or \
            kwargs['task_name'] == 'FastBEV_f4':
            class_to_name = class_to_name_type_1
        else:
            class_to_name = class_to_name_type_2
        det_classes = [class_to_name[i] for i in range(self.num_classes)]

        if kwargs['dataset_category'] == datasets.DATASET_CATEGORY_NUSCENES_FRAME:
            result_dict['pred_instances_3d'] = \
                self.format_results_lidar_bbox(predictions, det_classes, **kwargs)
        else:
            result_dict['pred_instances_3d'] = \
                self.format_results_camera_bbox(predictions, det_classes, **kwargs)


        metric_dict = {}
        for metric in self.metrics:
            ap_dict = self.nus_evaluate(
                result_dict, det_classes=det_classes, metric=metric)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        return metric_dict

    # Evaluation in Nuscenes protocol
    # https://github.com/open-mmlab/mmdetection3d/
    # Based on nus_evaluate()
    def nus_evaluate(self, 
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

        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = os.path.join(tmp_dir.name, 'results')
        os.makedirs(output_dir, exist_ok=True)

        nusc_eval = NuScenesEval(
            self.nusc,
            self.data_infos,
            self.data_scene_infos,
            pred_boxes,
            data_ids=self.data_ids,
            config=self.eval_detection_configs,
            eval_set='mini_val',
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main()

        # record metrics
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'r') as f:
            metrics = json.load(f)

        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in det_classes:
            for k, v in metrics['label_aps'][name].items():
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

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return detail


    # Format 3D LiDAR boxes
    # https://github.com/open-mmlab/mmdetection3d/
    # Based on _format_lidar_bbox() 
    def format_results_lidar_bbox(self, predictions, det_classes, **kwargs):
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(predictions):
            annos = []
            boxes, attrs = output_to_nusc_box(det, kwargs['task_name'], bbox3d_type='lidar')

            # sample_idx in sequential order
            sample_idx = i 
            sample_token = self.data_infos['infos'][sample_idx]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos['infos'][sample_idx],
                                             boxes, det_classes,
                                             self.eval_detection_configs,
                                             kwargs['task_name'])

            for i, box in enumerate(boxes):
                name = det_classes[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)

            nusc_annos[sample_token] = annos

        return nusc_annos


    # Get attribute from predicted index.
    # https://github.com/open-mmlab/mmdetection3d/
    # Based on get_attr_name() 
    def get_attr_name(self, attr_idx: int, label_name: str) -> str:
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one in the
        attribute set. If it is consistent with the category, we will keep it.
        Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
            'pedestrian.standing', 'pedestrian.sitting_lying_down',
            'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
        ]
        if label_name == 'car' or label_name == 'bus' \
            or label_name == 'truck' or label_name == 'trailer' \
                or label_name == 'construction_vehicle':
            if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
                AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
                    AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        elif label_name == 'pedestrian':
            if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
                AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
                    AttrMapping_rev2[attr_idx] == \
                    'pedestrian.sitting_lying_down':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        elif label_name == 'bicycle' or label_name == 'motorcycle':
            if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
                    AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
                return AttrMapping_rev2[attr_idx]
            else:
                return self.DefaultAttribute[label_name]
        else:
            return self.DefaultAttribute[label_name]


    # Format 3D LiDAR boxes
    # https://github.com/open-mmlab/mmdetection3d/
    # Based on _format_camera_bbox()
    def format_results_camera_bbox(self, predictions, det_classes, **kwargs):
        nusc_annos = {}

        print('\nStart to convert detection format...')

        # Camera types in Nuscenes datasets
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]

        CAM_NUM = 6

        for i, det in enumerate(predictions):
            # sample_idx in sequential order
            sample_idx = i
            frame_sample_idx = sample_idx // CAM_NUM
            camera_type_id   = sample_idx % CAM_NUM

            if camera_type_id == 0:
                boxes_per_frame = []
                attrs_per_frame = []

            annos = []
            boxes, attrs = output_to_nusc_box(det, kwargs['task_name'],
                                              bbox3d_type='camera')
            sample_token = self.data_infos['infos'][sample_idx]['token']
            camera_type = camera_types[camera_type_id]
            
            # Convert the NuScenesBox box from camera to global coordinate
            boxes, attrs = cam_nusc_box_to_global(self.data_infos['infos'][sample_idx],
                                           boxes, attrs, det_classes,
                                           self.eval_detection_configs,
                                           camera_type)

            boxes_per_frame.extend(boxes)
            attrs_per_frame.extend(attrs)

            # Remove redundant predictions caused by overlap of images
            if (sample_idx + 1) % CAM_NUM != 0:
                continue

            # Convert the NuScenesBox box from global to (FRONT) camera coordinate
            boxes = global_nusc_box_to_cam(self.data_infos['infos'][sample_idx],
                                           boxes_per_frame, det_classes,
                                           self.eval_detection_configs,
                                           camera_type=camera_type)

            # Convert boxes from NuScenesBox to tensor array from CameraInstance3DBoxes
            boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes)

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

            bev = np.copy(boxes3d[:, [0, 2, 3, 5, 6]])
            # positive direction of the gravity axis
            # in cam coord system points to the earth
            # so the bev yaw angle needs to be reversed
            bev[:, -1] = -bev[:, -1]
            boxes3d_for_nms = xywhr2xyxyr(bev)
            #boxes3d = cam_boxes3d.tensor

            # generate attr scores from attr labels
            attrs = np.array([attr for attr in attrs_per_frame], dtype=labels.dtype)
            boxes3d, scores, labels, attrs = box3d_multiclass_nms(
                boxes3d,
                boxes3d_for_nms,
                scores,
                nms_cfg['score_thr'],
                nms_cfg['max_per_frame'],
                nms_cfg,
                mlvl_attr_scores=attrs)

            #cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(boxes3d, scores, labels, attrs)
            boxes, attrs = output_to_nusc_box(det, kwargs['task_name'],
                                              bbox3d_type='camera')
            
            # Convert boxes from NuScenesBox in (FRONT) camera coor to global coordinate
            boxes, attrs = cam_nusc_box_to_global(
                self.data_infos['infos'][sample_idx], boxes, attrs, det_classes,
                self.eval_detection_configs, camera_type=camera_type, front_cam=True)

            for i, box in enumerate(boxes):
                name = det_classes[box.label]
                attr = self.get_attr_name(attrs[i], name)

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)

            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        return nusc_annos

    def evaluate(self, predictions, **kwargs):
        return 0

