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

# Also includes parts from: https://github.com/open-mmlab/mmpose
# License: https://github.com/open-mmlab/mmpose/blob/master/LICENSE
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

import numpy as np
from PIL import Image
from . import functional as F

import math
import copy

from scipy.spatial import transform
from pyquaternion import Quaternion
from ..utils.config_utils.misc_utils import inverse_sigmoid as inverse_sigmoid


# pulled exactly from the nuscenes official code for installation on evm
# from nuscenes.eval.common.utils import quaternion_yaw 
def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


_camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
]

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

class ImageCrop():
    """Crops the given image at the specified region
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired area to crop. 
    """

    def __init__(self, dim=None):
        super().__init__()
        if dim is None:
            self.dim = dim
        else:
            if len(dim) != 4:
                raise ValueError("Please provide (top, left, height, width) for cropping area.")
            self.dim = dim
        #

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.dim is not None:
            if isinstance(img, list):
                for i in range(len(img)):
                    img[i] = F.crop(img[i], self.dim[1], self.dim[0], self.dim[3], self.dim[2])
            else:
                img = F.crop(img, self.dim[1], self.dim[0], self.dim[3], self.dim[2])

        return img, info_dict

    def __repr__(self):
        if self.dim is not None:
            return self.__class__.__name__ + '(dim={0})'.format(self.dim)
        else:
            return self.__class__.__name__ + '()'


class ImagePad():
    """Pad the given image at the specified paded image size
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired size to pad. 
    """

    def __init__(self, dim=None):#, pad_color=(103.530, 116.280, 123.675)):
        super().__init__()
        if dim is None:
            self.dim = dim
        else:
            if len(dim) != 4:
                raise ValueError("Please provide (left, top, right, bottom) to pad")
            self.dim = dim
        #self.pad_color = pad_color

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        if self.dim is not None:
            if isinstance(img, list):
                for i in range(len(img)):
                    img[i] = F.pad(img[i], self.dim)
                info_dict['pad_shape'] = (img[0].shape[0], img[0].shape[1])
            else:
                img = F.pad(img, self.dim)

                #img = img.astype(np.float32)
                ## left and top
                #if self.dim[0] != 0 or self.dim[1] !=0:
                #    img[:self.dim[1], :self.dim[0], 0] = self.pad_color[0]
                #    img[:self.dim[1], :self.dim[0], 1] = self.pad_color[1]
                #    img[:self.dim[1], :self.dim[0], 2] = self.pad_color[2]
                ## right and bottom
                #if self.dim[2] != 0 or self.dim[3] !=0:
                #    img[-self.dim[3]:, -self.dim[2]:, 0] = self.pad_color[0]
                #    img[-self.dim[3]:, -self.dim[2]:, 1] = self.pad_color[1]
                #    img[-self.dim[3]:, -self.dim[2]:, 2] = self.pad_color[2]
                info_dict['pad_shape'] = (img.shape[0], img.shape[1])
        return img, info_dict

    def __repr__(self):
        if self.dim is not None:
            return self.__class__.__name__ + '(dim={0})'.format(self.dim)
        else:
            return self.__class__.__name__ + '()'



class BEVSensorsRead():

    """ Set sensor (image) sizes for BEV network, and 
        initialize and update camera intrinsic/extrinsic parameters.

        Args:
            imsize, Tensor: source image size.
            resize, Tensor: image size after resizing iamge.
            crop, Tensor: croped area of resized image. It coulde be larger than 
                          resized image when resized image should be padded.
            load type, str: 'frame_based' for BEV, 'mv_image_based' for a single frame detection

        Returns:
            Sensor (image) file name list.
            Dictionary (info_dict) that has all necessary parameters.

    """
    def __init__(self, imsize, resize, crop, load_type='frame_based'):
        self.load_type = load_type
        self.camera_types = _camera_types
        self.imsize = imsize
        self.resize = resize
        self.crop = crop

        self.prev_frame_info = {
            'prev_bev_exist': False,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def get_sensor_transforms(self, data, cam_name):
        w, x, y, z = data['cams'][cam_name]['sensor2ego_rotation']
        
        # sweep sensor to sweep ego
        sensor2ego_rot = np.array(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = np.array(
            data['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = np.zeros((4,4))  # sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        
        # sweep ego to global
        w, x, y, z = data['ego2global_rotation']
        ego2global_rot = np.array(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = np.array(data['ego2global_translation'])
        ego2global = np.zeros((4,4)) # ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran

        return sensor2ego, ego2global


    def get_calib_data(self, data, info_dict):
        # For camera transforms
        intrins = []
        post_intrins = []
        sensor2egos = []
        ego2globals = []
        post_rots = []
        post_trans = []
        lidar2cams = []
        cam2lidars = []
        lidar2imgs = []
        lidar2imgs_org = []
        ego2imgs = []
        
        for cam_name,dic in data['cams'].items():
            intrin    = np.array(dic['cam_intrinsic']).astype(np.float32)
            lidar2cam = np.array(dic['lidar2sensor']).astype(np.float32)
            cam2lidar = np.linalg.inv(lidar2cam)

            sensor2ego, ego2global = \
                self.get_sensor_transforms(data, cam_name)
            
            post_rot = np.eye(3)
            post_tran = np.zeros(3)

            # imgsize is from info_dict
            scale = self.resize[1] / self.imsize[1]
            post_rot[:2, :2] *= scale
            post_tran[:2] -= np.array(self.crop[:2])

            # camera instrinsic after resizing and cropping
            temp = copy.deepcopy(post_rot)
            temp[:2, 2] = post_tran[:2]
            post_intrin = (temp @ intrin).astype(np.float32)

            # lidar2img transform after resizing and cropping
            lidar2img = np.eye(4)
            lidar2img[:3, :3] = post_intrin
            lidar2img = (lidar2img @ lidar2cam).astype(np.float32)

            # lidar2img_org before resizing and cropping
            # Needed for visualization
            lidar2img_org = np.eye(4)
            lidar2img_org[:3, :3] = intrin
            lidar2img_org = (lidar2img_org @ lidar2cam).astype(np.float32)

            # ego2img
            # Needed for visualization of BEVDet
            ego2img = np.eye(4)
            ego2img[:3, :3] = intrin
            ego2img = (ego2img @ np.linalg.inv(sensor2ego)).astype(np.float32)

            intrins.append(intrin)
            post_intrins.append(post_intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            lidar2cams.append(lidar2cam)
            cam2lidars.append(cam2lidar)
            lidar2imgs.append(lidar2img)
            lidar2imgs_org.append(lidar2img_org)
            ego2imgs.append(ego2img)

        # For BEV transform
        bda_mat = np.zeros((4, 4))
        bda_mat[0, 0] = bda_mat[1, 1] = bda_mat[2, 2] = bda_mat[3, 3] = 1
        
        # expand array assuming batch_size = 1
        info_dict['num_cams']       = len(data['cams'])
        info_dict['intrins']        = np.expand_dims(np.stack(intrins), 0)
        info_dict['post_intrins']   = np.expand_dims(np.stack(post_intrins), 0)
        info_dict['sensor2egos']    = np.expand_dims(np.stack(sensor2egos), 0)
        info_dict['ego2globals']    = np.expand_dims(np.stack(ego2globals), 0)
        info_dict['post_rots']      = np.expand_dims(np.stack(post_rots), 0)
        info_dict['post_trans']     = np.expand_dims(np.stack(post_trans), 0)
        info_dict['lidar2cams']     = lidar2cams # np.expand_dims(np.stack(lidar2cams), 0)
        info_dict['cam2lidars']     = cam2lidars
        info_dict['lidar2imgs']     = lidar2imgs # np.expand_dims(np.stack(lidar2imgs), 0)
        info_dict['lidar2imgs_org'] = lidar2imgs_org
        info_dict['ego2imgs']       = ego2imgs
        info_dict['bda']            = np.expand_dims(bda_mat, 0)
        info_dict['pad_shape']      = (self.crop[3], self.crop[2])
        info_dict['lidar2ego']      = np.array(data['lidar2ego'])
        info_dict['scene_token']    = data['scene_token']


        if info_dict['task_name'] == 'BEVFormer':
            info_dict['prev_bev_exist'] = True
            if info_dict['scene_token'] != self.prev_frame_info['scene_token']:
                info_dict['prev_bev_exist'] = False

            # can_bus
            matrot = ego2globals[0]
            rotation = transform.Rotation.from_matrix(matrot[:3, :3]).as_quat()
            rotation = Quaternion(a=rotation[3], i=rotation[0], j=rotation[1], k=rotation[2])

            can_bus = data['can_bus']
            can_bus[:3] = matrot[:3, 3]
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

            tmp_pos = copy.deepcopy(data['can_bus'][:3])
            tmp_angle = copy.deepcopy(data['can_bus'][-1])
            if info_dict['prev_bev_exist'] == True:
                can_bus[:3] -= self.prev_frame_info['prev_pos']
                can_bus[-1] -= self.prev_frame_info['prev_angle']
            else:
                can_bus[:3] = 0
                can_bus[-1] = 0

            info_dict['can_bus'] = can_bus

            self.prev_frame_info['scene_token'] = info_dict['scene_token']
            self.prev_frame_info['prev_pos']    = tmp_pos
            self.prev_frame_info['prev_angle']  = tmp_angle

        return info_dict

    def __call__(self, data, info_dict):
        if self.load_type == 'mv_image_based':
            info_dict['camera_type'] = data['camera_type']
            info_dict['ego2global']  = data['ego2global']
            info_dict['timestamp']   = data['timestamp']
            info_dict['intrins']     = data['images'][data['camera_type']]['cam_intrinsic']

            image_name = data['images'][data['camera_type']]['data_path']
            return image_name, info_dict
        else:
            image_name_list = []
            for cam, dic in data['cams'].items():
                image_name_list.append(dic['data_path'])

            # save lidar_path, which is also needed for visualization
            info_dict['lidar_path'] = data['lidar_path']
            info_dict = self.get_calib_data(data, info_dict)

            return tuple(image_name_list), info_dict


class GetPETRGeometry():
    def __init__(self, crop, featsize):
        # Params needed to generate coords3d: How make them configurable?
        # Batch size
        self.B = 1
        self.C              = 256
        self.H              = featsize[0]
        self.W              = featsize[1]

        self.position_level = 0
        self.with_multiview = True
        self.LID            = True
        self.depth_num      = 64
        self.depth_start    = 1
        self.position_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]


    def create_coords3d(self, info_dict):
        batch_size = self.B
        num_cams = info_dict['num_cams']
        pad_h, pad_w = info_dict['pad_shape']

        # forward() in petr_head.py
        # masks is simply (B, N, self.H, self.W) array initialized to False
        masks = np.zeros((batch_size, num_cams, self.H, self.W)).astype(np.bool)

        eps = 1e-5
        B, N, C, H, W = self.B, num_cams, self.C, self.H, self.W
        coords_h = np.arange(H).astype(np.float32) * pad_h / H
        coords_w = np.arange(W).astype(np.float32) * pad_w / W

        if self.LID:
            index = np.arange(
                start=0,
                stop=self.depth_num,
                step=1).astype(np.float32)
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = np.arange(
                start=0,
                stop=self.depth_num,
                step=1).astype(np.float32)
            bin_size = (self.position_range[3] -
                        self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = np.transpose(np.stack(np.meshgrid(coords_w, coords_h, coords_d)), (2, 1, 3, 0))  # W, H, D, 3
        coords = np.concatenate((coords, np.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * np.maximum(
            coords[..., 2:3], np.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        img2lidar = []
        for i in range(len(info_dict['lidar2imgs'])):
            img2lidar.append(np.linalg.inv(info_dict['lidar2imgs'][i]))
        img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        
        #img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = np.repeat(coords.reshape(1, 1, W, H, D, 4, 1), 6, 1)
        #img2lidars = img2lidars.reshape(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        img2lidars = np.repeat(img2lidars.reshape(B, N, 1, 1, 1, 4, 4), W, 2)
        img2lidars = np.repeat(img2lidars, H, 3)
        img2lidars = np.repeat(img2lidars, D, 4)
        coords3d = np.squeeze(np.matmul(img2lidars, coords), -1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        #coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = coords_mask.reshape(B, N, W, H, D*3).sum(-1) > (D*0.5)
        coords_mask = masks | coords_mask.transpose(0, 1, 3, 2)
        coords3d = np.ascontiguousarray(coords3d.transpose(0, 1, 4, 5, 3, 2)).reshape(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)

        return masks, coords3d


    def __call__(self, data, info_dict):
        masks, coords3d = self.create_coords3d(info_dict)

        ## for petr, combine all 6 images into on
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        ## append coords3d. masks are not needed.
        data.append(coords3d)

        return data, info_dict

class GetBEVDetGeometry():

    def __init__(self, crop):
        # how to configure these params?
        self.crop = crop
        self.downsample = 16
        self.grid_config = {
            'x': [-51.2, 51.2, 0.8],
            'y': [-51.2, 51.2, 0.8],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 1.0],
        }
        self.out_channels = 64

        self.create_grid_infos(**self.grid_config)

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = np.array([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = np.array([cfg[2] for cfg in [x, y, z]])
        self.grid_size = np.array([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, info_dict):
        h_in, w_in = self.crop[3], self.crop[2]
        h_feat, w_feat = h_in // self.downsample, w_in // self.downsample

        depth_cfg = self.grid_config['depth']

        #d = np.arange(*depth_cfg, dtype=float) \
        #    .view(-1, 1, 1).expand(-1, h_feat, w_feat)
        d = np.arange(*depth_cfg, dtype=np.float32).reshape(-1, 1, 1)
        d = np.broadcast_to(d, (d.shape[0], h_feat, w_feat))
        depth_channels = d.shape[0]


        #x = np.linspace(0, w_in - 1, w_feat,  dtype=float)\
        #    .view(1, 1, w_feat).expand(depth_channels, h_feat, w_feat)
        #y = np.linspace(0, h_in - 1, h_feat,  dtype=float)\
        #    .view(1, h_feat, 1).expand(depth_channels, h_feat, w_feat)

        x = np.linspace(0, w_in - 1, w_feat, dtype=np.float32).reshape(1, 1, w_feat)
        x = np.broadcast_to(x, (depth_channels, h_feat, w_feat))
        y = np.linspace(0, h_in - 1, h_feat, dtype=np.float32).reshape(1, h_feat, 1)
        y = np.broadcast_to(y, (depth_channels, h_feat, w_feat))

        # D x H x W x 3
        return np.stack((x, y, d), -1)

    def get_lidar_coor(self, info_dict):

        cam2imgs    = info_dict['intrins']
        sensor2egos = info_dict['sensor2egos']
        post_rots   = info_dict['post_rots']
        post_trans  = info_dict['post_trans']
        bda         = info_dict['bda']

        frustum = self.create_frustum(info_dict)

        B, N, _, _ = sensor2egos.shape
        #N = sensor2egos.shape[0]

        # post-transformation
        # B x N x D x H x W x 3
        points = frustum - post_trans.reshape(B, N, 1, 1, 1, 3)
        #points = np.linalg.inv(post_rots).reshape(B, N, 1, 1, 1, 3, 3) \
        #    .matmul(points.unsqueeze(-1))
        points = np.matmul(np.linalg.inv(post_rots).reshape(B, N, 1, 1, 1, 3, 3), \
            np.expand_dims(points, -1))

        # cam_to_ego
        points = np.concatenate(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        #combine = sensor2egos[:,:,:3,:3].matmul(np.linalg.inv(cam2imgs))
        #points = combine.reshape(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        combine = np.matmul(sensor2egos[:,:,:3,:3], np.linalg.inv(cam2imgs))
        points = np.squeeze(np.matmul(combine.reshape(B, N, 1, 1, 1, 3, 3), points), -1)
        points += sensor2egos[:,:,:3, 3].reshape(B, N, 1, 1, 1, 3)
        #points = bda[:, :3, :3].reshape(B, 1, 1, 1, 1, 3, 3).matmul(
        #    points.unsqueeze(-1)).squeeze(-1)
        points = np.squeeze(np.matmul(bda[:, :3, :3].reshape(B, 1, 1, 1, 1, 3, 3), \
             np.expand_dims(points, -1)), -1)
        points += bda[:, :3, 3].reshape(B, 1, 1, 1, 1, 3)

        return points

    def precompute_voxel_info(self, coor):
        B, N, D, H, W, _ = coor.shape

        num_points = B * N * D * H * W

        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound) /
                self.grid_interval)

        coor = coor.astype(np.long).reshape(num_points, 3)
        #batch_idx = np.arange(0, B).reshape(B, 1). \
        #    expand(B, num_points // B).reshape(num_points, 1)
        batch_idx = np.arange(0, B).reshape(B, 1)
        batch_idx = np.broadcast_to(batch_idx, (B, num_points // B)).reshape(num_points, 1)
        coor = np.concatenate((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])

        # for our BEV pooling - coor in 1D tensor
        num_grids = (B*self.grid_size[2]*self.grid_size[1]*self.grid_size[0]).astype(np.int)

        #bev_feat = np.zeros((num_grids + 1, self.out_channels), device=coor.device)
        bev_feat = np.zeros((num_grids + 1, self.out_channels), dtype=np.float32)

        coor_1d = np.zeros(num_points)
        coor_1d  = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) + \
                   coor[:, 2] * (self.grid_size[1] * self.grid_size[0]) + \
                   coor[:, 1] *  self.grid_size[0] + coor[:, 0]
        coor_1d[np.where(kept==False)] = (B * self.grid_size[2] * self.grid_size[1] * self.grid_size[0]).astype(np.long)
        #for i in range(num_points):
        #    if kept[i]:
        #        coor_1d[i]  = coor[i, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) + \
        #                     coor[i, 2] * (self.grid_size[1] * self.grid_size[0]) + \
        #                     coor[i, 1] *  self.grid_size[0] + coor[:, 0]
        #    else:
        #        coor_1d[i] = B * self.grid_size[2] * self.grid_size[1] * self.grid_size[0]

        return bev_feat, np.ascontiguousarray(coor_1d.astype(np.long))


    def __call__(self, data, info_dict):
        coor = self.get_lidar_coor(info_dict)
        bev_feat, lidar_coor_1d = self.precompute_voxel_info(coor)

        ##  combine all 6 images into on
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        # append bev_feat and lidar_coor_1d
        data.append(bev_feat)
        data.append(lidar_coor_1d)

        return data, info_dict


class GetBEVFormerGeometry():

    def __init__(self, crop):
        # how to configure these params?
        self.bev_h = 50
        self.bev_w = 50
        self.num_points_in_pillar = 4
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.real_h = 102.4
        self.real_w = 102.4

        self.rotate_prev_bev =  True
        self.rotate_center = [100, 100]

    def get_reference_points(self, H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, dtype=np.float32):
        """
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            NP array: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = np.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype).reshape(-1, 1, 1)
            zs = np.broadcast_to(zs, (num_points_in_pillar, H, W)) / Z

            xs = np.linspace(0.5, W - 0.5, W, dtype=dtype).reshape(1, 1, W)
            xs = np.broadcast_to(xs, (num_points_in_pillar, H, W)) / W

            ys = np.linspace(0.5, H - 0.5, H, dtype=dtype).reshape(1, H, 1)
            ys = np.broadcast_to(ys, (num_points_in_pillar, H, W)) / H

            ref_3d = np.stack((xs, ys, zs), -1)

            ref_3d = np.transpose(ref_3d, (0, 3, 1, 2))
            B, C, H, W = ref_3d.shape
            ref_3d = np.transpose(ref_3d.reshape(B, C, H*W), (0, 2, 1))
            ref_3d = np.repeat(ref_3d[None], bs, 0)

            return ref_3d

    def point_sampling(self, reference_points, pc_range,  info_dict):

        lidar2img = []
        lidar2img.append(info_dict['lidar2imgs'])
        lidar2img = np.asarray(lidar2img)
        #lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.copy()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = np.concatenate(
            (reference_points, np.ones_like(reference_points[..., :1])), -1)

        reference_points = np.transpose(reference_points, (1, 0, 2, 3)) # 4x1x2500x4
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        reference_points = reference_points.reshape(D, B, 1, num_query, 4)
        reference_points = np.expand_dims(np.repeat(reference_points, num_cam, 2), -1)

        lidar2img = lidar2img.reshape(1, B, num_cam, 1, 4, 4)
        lidar2img = np.repeat(lidar2img, D, 0)
        lidar2img = np.repeat(lidar2img, num_query, 3)

        reference_points_cam = np.squeeze(np.matmul(lidar2img.astype(np.float32),
                                            reference_points.astype(np.float32)), -1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / np.maximum(
            reference_points_cam[..., 2:3], np.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= info_dict['pad_shape'][1]
        reference_points_cam[..., 1] /= info_dict['pad_shape'][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        bev_mask = np.nan_to_num(bev_mask)

        reference_points_cam = np.transpose(reference_points_cam, (2, 1, 3, 0, 4))
        bev_mask = np.squeeze(np.transpose(bev_mask, (2, 1, 3, 0, 4)), -1)

        return reference_points_cam, bev_mask


    def precompute_bev_info(self, info_dict):

        # Pre-compute the voxel info 
        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar,
            dim='3d', bs=1, dtype=np.float32)

        # Get image coors corresponding to ref_3d. bev_mask indicates valid coors
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, info_dict)

        bev_valid_indices = []
        bev_valid_indices_count = []
        for mask_per_img in bev_mask:
            #index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            nzindex = np.squeeze(np.nonzero(np.sum(mask_per_img[0], -1))[0])
            index_query_per_img = np.ones(self.bev_h*self.bev_w) * self.bev_h*self.bev_w
            index_query_per_img[:len(nzindex)] = nzindex
            bev_valid_indices.append(index_query_per_img)
            bev_valid_indices_count.append(len(nzindex))

        # Get bev_mask_count from bev_mask for encoder spatial_cross_attention
        bev_mask_count = np.sum(bev_mask, -1) > 0
        bev_mask_count = np.sum(np.transpose(bev_mask_count, (1, 2, 0)), -1)
        bev_mask_count = np.clip(bev_mask_count, a_min=1.0, a_max=None).astype(np.float32)
        bev_mask_count = bev_mask_count[..., None]

        can_bus = np.expand_dims(info_dict['can_bus'], 0)

        delta_x = np.array([info_dict['can_bus'][0]])
        delta_y = np.array([info_dict['can_bus'][1]])
        ego_angle = np.array([info_dict['can_bus'][-2] / np.pi * 180])
        grid_length_y = self.real_h / self.bev_h
        grid_length_x = self.real_w / self.bev_w
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w

        return reference_points_cam, bev_mask_count, \
               np.expand_dims(np.concatenate(bev_valid_indices, axis=0), axis=1).astype(np.int32), \
               np.array(bev_valid_indices_count).astype(np.int32), \
               np.array([[shift_x[0],shift_y[0]]]).astype(np.float32), can_bus.astype(np.float32)


    # Based on torchvision.transforms.functional._get_inverse_affine_matrix()
    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear, inverted=True):
        # Helper method to compute inverse matrix for affine transformation

        # Pillow requires inverse affine transformation matrix:
        # Affine matrix is : M = T * C * RotateScaleShear * C^-1
        #
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RotateScaleShear is rotation with scale and shear matrix
        #
        #       RotateScaleShear(a, s, (sx, sy)) =
        #       = R(a) * S(s) * SHy(sy) * SHx(sx)
        #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
        #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
        #         [ 0                    , 0                                      , 1 ]
        # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
        # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
        #          [0, 1      ]              [-tan(s), 1]
        #
        # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1
        rot = math.radians(angle)
        sx = math.radians(shear[0])
        sy = math.radians(shear[1])

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        if inverted:
            # Inverted rotation matrix with scale and shear
            # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
            matrix = [d, -b, 0.0, -c, a, 0.0]
            matrix = [x / scale for x in matrix]
            # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
            matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
            matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
            # Apply center translation: C * RSS^-1 * C^-1 * T^-1
            matrix[2] += cx
            matrix[5] += cy
        else:
            matrix = [a, b, 0.0, c, d, 0.0]
            matrix = [x * scale for x in matrix]
            # Apply inverse of center translation: RSS * C^-1
            matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
            matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
            # Apply translation and center : T * C * RSS * C^-1
            matrix[2] += cx + tx
            matrix[5] += cy + ty

        return matrix


    # Based on torchvision.transforms._functional_tensor._gen_affine_grid()
    def gen_affine_grid(self, theta, w, h, ow, oh):
        # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
        # AffineGridGenerator.cpp#L18
        # Difference with AffineGridGenerator is that:
        # 1) we normalize grid values after applying theta
        # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

        d = 0.5
        base_grid = np.empty([1, oh, ow, 3], dtype=theta.dtype)
        x_grid = np.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)
        base_grid[..., 0] = np.copy(x_grid)
        y_grid = np.expand_dims(np.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh), -1)
        base_grid[..., 1] = np.copy(y_grid)
        base_grid[..., 2].fill(1)

        rescaled_theta = np.transpose(theta, (0, 2, 1)) / np.array([0.5 * w, 0.5 * h], dtype=theta.dtype)
        output_grid = np.matmul(base_grid.reshape(1, oh * ow, 3), rescaled_theta)
        return output_grid.reshape(1, oh, ow, 2)


    def compute_rotation_matrix(self, info_dict):
        height = self.bev_h
        width  = self.bev_w
        oh = height
        ow = width
        dtype = np.float32

        center_f = [0.0, 0.0]
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(self.rotate_center, [width, height])]

        angle = info_dict['can_bus'][-1]
        matrix = self.get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

        theta = np.array(matrix, dtype=dtype).reshape(1, 2, 3)
        grid = self.gen_affine_grid(theta, width, height, ow, oh)
        return grid


    def __call__(self, data, info_dict):
        reference_points_cam, bev_mask_count, bev_valid_indices, bev_valid_indices_count, shift_yx, can_bus = \
            self.precompute_bev_info(info_dict)

        rotation_grid = self.compute_rotation_matrix(info_dict)

        ## for bevformer, combine all 6 images into on
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        data.append(shift_yx)
        data.append(rotation_grid)
        data.append(reference_points_cam)
        data.append(bev_mask_count)
        data.append(bev_valid_indices)
        # Not needed for the latest model (bevformer_tiny_plus_480x800_20250408.onnx)
        #data.append(bev_valid_indices_count)
        data.append(can_bus)

        return data, info_dict

class GetFCOS3DGeometry():
    def __init__(self):
        pass

    def __call__(self, img, info_dict):
        cam2img = info_dict['intrins']
        pad_cam2img = np.eye(4, dtype=np.float32)
        pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
        inv_pad_cam2img = np.linalg.inv(pad_cam2img).transpose(1, 0)

        data = []
        data.append(img)
        data.append(pad_cam2img)
        data.append(inv_pad_cam2img)

        return data, info_dict


class GetFastBEVGeometry():
    """ Create input data including camera frustum (i.e. 3D volume around ego vehicle)
        which is needed to run FastBEV. This camera frustum is constructed based on 
        cameras' intrinsic/extrinsic params.
    """
    def __init__(self, crop):
        # how to configure these params?
        self.feats_size        = [6, 64, 64, 176]
        self.n_voxels          = [200, 200, 4]
        self.voxel_size        = [0.5, 0.5, 1.5]
        self.point_cloud_range = [-50, -50, -5, 50, 50, 3]

        #self.crop = crop
        #self.downsample = 16
        #self.grid_config = {
        #    'x': [-51.2, 51.2, 0.8],
        #    'y': [-51.2, 51.2, 0.8],
        #    'z': [-5, 3, 8],
        #    'depth': [1.0, 60.0, 1.0],
        #}
        #self.out_channels = 64

        #self.create_grid_infos(**self.grid_config)

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = np.eye(3)
        intrinsic[:2] /= stride
        for cam_id in range(len(img_meta['lidar2imgs'])):
            extrinsic = img_meta['lidar2imgs'][cam_id]
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])

        return np.stack(projection)

    @staticmethod
    def get_points(n_voxels, voxel_size, origin):
        points = np.stack(
            np.meshgrid(np.arange(n_voxels[0]),
                        np.arange(n_voxels[1]),
                        np.arange(n_voxels[2]), indexing='ij')
        )
        new_origin = origin - n_voxels / 2.0 * voxel_size
        points = points * voxel_size.reshape(3, 1, 1, 1) + new_origin.reshape(3, 1, 1, 1)
        return points

    @staticmethod
    def get_augmented_img_params(img_meta):
        fH, fW  = img_meta['pad_shape']
        H, W, _ = img_meta['data_shape']

        resize = float(fW)/float(W)
        resize_dims = (int(W * resize), int(H * resize))

        newW, newH = resize_dims
        crop_h_start = (newH - fH) // 2
        crop_w_start = (newW - fW) // 2
        crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

        return resize, resize_dims, crop

    @staticmethod
    def scale_augmented_img_params(post_rot, post_tran, resize_r, resize_dims, crop):
        post_rot *= resize_r
        post_tran -= np.array(crop[:2])

        A = get_rot(0)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        post_rot = np.matmul(A, post_rot)
        post_tran = np.matmul(A, post_tran) + b

        ret_post_rot, ret_post_tran = np.eye(3), np.zeros(3)
        ret_post_rot[:2, :2] = post_rot
        ret_post_tran[:2] = post_tran

        return ret_post_rot, ret_post_tran

    def rts2proj(self, img_meta, post_rot=None, post_tran=None):
        if img_meta is None:
            return None

        for cam_id in range(len(img_meta['lidar2imgs'])):
            lidar2cam = img_meta['lidar2cams'][cam_id]
            intrinsic = img_meta['intrins'][0][cam_id]

            viewpad = np.eye(4)
            if post_rot is not None:
                assert post_tran is not None, [post_rot, post_tran]
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = post_rot @ intrinsic
                viewpad[:3, 2] += post_tran
            else:
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            img_meta['lidar2imgs'][cam_id] = (viewpad @ lidar2cam).astype(np.float32)

        return img_meta

    def precompute_volume_info(self, points, projection):
        """
        function: 2d feature + predefined point cloud -> 3d volume
        """
        n_images, n_channels, height, width = self.feats_size
        n_x, n_y, n_z = points.shape[-3:]

        points = np.broadcast_to(points.reshape(1, 3, -1), (n_images, 3, n_x*n_y*n_z))
        points = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
    
        # ego_to_cam
        points_2d_3 = np.matmul(projection, points)  # lidar2img
        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().astype(np.long)  # [6, 160000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().astype(np.long)  # [6, 160000]
        z = points_2d_3[:, 2]  # [6, 160000]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 160000]

        # xy coordinate
        xy_coor = y * width + x

        coor      = np.full((1, xy_coor.shape[1]), width*height*n_images)
        cum_valid = np.full((1, xy_coor.shape[1]), True)
        cum_valid = cum_valid[0]

        for i in reversed(range(n_images)):
            valid_idx = np.multiply(cum_valid, valid[i]).astype(np.bool)
            coor[0, valid_idx] = xy_coor[i, valid_idx] + i*width*height
            cum_valid[valid_idx] = False

        return coor[0]

    def precompute_proj_info(self, data, info_dict, prev_img_metas=None):
        xy_coor_list   = []

        n_times = 1
        if 'num_bev_temporal_frames' in info_dict:
            n_times = info_dict['num_bev_temporal_frames'] + 1
        stride_i = math.ceil(data[0].shape[-1] / self.feats_size[-1])

        for batch_id in range(len(info_dict['intrins'])):
            img_meta_list = []
            img_meta = copy.deepcopy(info_dict)

            if isinstance(img_meta["pad_shape"], list):
                img_meta["pad_shape"] = img_meta["pad_shape"][0]

            # add to img_meta_list
            img_meta_list.append(img_meta)

            # update adjacent img_metas:
            #  refer to get_adj_data_info() and refine_data_info()
            #  in projects/FastBEV/fast_bev/nuscenes_dataset.py
            for i in range(n_times - 1):
                prev_img_meta = copy.deepcopy(prev_img_metas[i])

                egocurr2global = img_meta['ego2globals'][batch_id][0]
                egoadj2global = prev_img_meta['ego2globals'][batch_id][0]
                lidar2ego = img_meta['lidar2ego']
                lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                    @ egoadj2global @ lidar2ego

                for cam_id in range(len(img_meta['lidar2imgs'])):
                    mat = lidaradj2lidarcurr @ img_meta['cam2lidars'][cam_id]
                    prev_img_meta['cam2lidars'][cam_id] = mat
                    prev_img_meta['lidar2cams'][cam_id] = np.linalg.inv(mat)

                    # obtain lidar to image transformation matrix
                    prev_img_meta['intrins'][batch_id][cam_id] = \
                        img_meta['intrins'][batch_id][cam_id]
                    intrin = prev_img_meta['intrins'][batch_id][cam_id]
                    viewpad = np.eye(4)
                    viewpad[:intrin.shape[0], :intrin.shape[1]] = intrin
                    prev_img_meta['lidar2imgs'][cam_id] = \
                        viewpad @ img_meta['lidar2cams'][cam_id]

                # Get augmented scaled params (lidar2imgs):
                resize_r, resize_dims, crop = self.get_augmented_img_params(prev_img_meta)
                post_rot, post_tran = self.scale_augmented_img_params(
                    np.eye(2), np.zeros(2),
                    resize_r=resize_r,
                    resize_dims=resize_dims,
                    crop=crop
                )

                prev_img_meta = self.rts2proj(prev_img_meta, post_rot, post_tran)

                if isinstance(prev_img_meta["pad_shape"], list):
                    prev_img_meta["pad_shape"] = prev_img_meta["pad_shape"][0]

                # add to img_meta_list
                img_meta_list.append(prev_img_meta)

            # precompute projection
            for seq_id in range(n_times):
                img_meta = img_meta_list[seq_id]

                projection = self._compute_projection(img_meta, stride_i, noise=0)

                # self.style in ['v1', 'v2']:
                n_voxels, voxel_size = self.n_voxels, self.voxel_size
                origin = (np.array(self.point_cloud_range[:3]) + 
                          np.array(self.point_cloud_range[3:])) / 2

                points = self.get_points(n_voxels=np.array(n_voxels),
                                         voxel_size=np.array(voxel_size),
                                         origin=origin)
                xy_coor = self.precompute_volume_info(points, projection).astype(np.int32)
                xy_coor_list.append(xy_coor)

        if n_times > 1:
            return np.stack(xy_coor_list)
        else:
            return xy_coor_list[0]


    def get_temporal_feats(self, info_dict):
        prev_feat = None
        prev_img_meta = info_dict
        prev_feats = []
        prev_img_metas = []

        if 'queue' in info_dict:
            num_prevs   = info_dict['num_bev_temporal_frames']
            queue_mem   = copy.deepcopy(info_dict['queue_mem'])
            feats_queue = info_dict['queue']
            del info_dict['queue_mem']
            del info_dict['queue']

            # Support only batch_size = 1s
            for i in range(1, num_prevs+1):
                cur_sample_idx = info_dict['sample_idx']

                if i > feats_queue.qsize() or \
                    info_dict['scene_token'] != queue_mem[cur_sample_idx - i]['img_meta']['scene_token']:
                    if prev_feat is None:
                        #prev_feats.append(np.zeros(self.feats_size, dtype=img.dtype))
                        prev_feats.append(np.zeros(self.feats_size, dtype=np.float32))
                        prev_img_metas.append(prev_img_meta)
                    else:
                        prev_feats.append(prev_feat)
                        prev_img_metas.append(prev_img_meta)
                else:
                    prev_feat = queue_mem[cur_sample_idx - i]['feature_map']
                    prev_img_meta = queue_mem[cur_sample_idx - i]['img_meta']
                    prev_feats.append(prev_feat)
                    prev_img_metas.append(prev_img_meta)

        return np.concatenate(prev_feats, axis=0), prev_img_metas

    def __call__(self, data, info_dict):

        # get previous temporal infos
        prev_feats_map = None
        prev_input_metas = None

        if 'num_bev_temporal_frames' in info_dict and info_dict['num_bev_temporal_frames'] > 0:
            prev_feats_map, prev_input_metas = self.get_temporal_feats(info_dict)
        xy_coors = self.precompute_proj_info(data, info_dict, prev_img_metas=prev_input_metas)

        ## for fastbev, combine all 6 images into one
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        data.append(xy_coors)
        if 'num_bev_temporal_frames' in info_dict and info_dict['num_bev_temporal_frames'] > 0:
            data.append(prev_feats_map)

        return data, info_dict
