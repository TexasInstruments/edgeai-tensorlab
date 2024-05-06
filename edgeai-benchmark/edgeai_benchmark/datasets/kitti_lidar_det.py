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
from .kitti_object_eval_python import kitti_common as kitti
from .kitti_object_eval_python.eval import get_official_eval_result, get_coco_eval_result
import numpy as np
import cv2

class KittiLidar3D(DatasetBase):
    def __init__(self, download=False, read_anno=True, dest_dir=None, num_frames=None, name='kitti_lidar_det', **kwargs):
        super().__init__(num_frames=num_frames, name=name, read_anno=read_anno, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in self.kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'

        path = self.kwargs['path']
        split_folder = self.kwargs['split']

        # download the data if needed
        if download:
            assert False , 'Download is not supported for this dataset'

        assert os.path.exists(path) and os.path.isdir(path), \
            utils.log_color('\nERROR', 'dataset path is empty', path)

        # create list of images and classes
        self.val_image_ids = utils.get_data_list(input= path + "/ImageSets/val.txt", dest_dir=dest_dir)
        self.num_frames = self.kwargs['num_frames'] = self.kwargs.get('num_frames',len(self.val_image_ids))
        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.val_image_ids)

        self.num_classes = kwargs['num_classes']

        if self.num_classes == 1:
            self.class_to_name = {
                                    0: 'Car'
                            }
        else:
            self.class_to_name = {
                                    0: 'Pedestrian',
                                    1: 'Cyclist',
                                    2: 'Car'
                            }

        self.class_names =[self.class_to_name[i] for i in range(self.num_classes)]

        self.pcd_limit_range = [0, -39.68, -3, 69.12, 39.68, 1]

        self.data_infos = []

        if read_anno:
            for idx in self.val_image_ids[:self.num_frames]:
                self.data_infos.append(self.read_db_info(idx))

            self.gt_annos = kitti.get_label_annos(os.path.join(self.kwargs['path'],self.kwargs['split'],'label_2'), self.val_image_ids[:self.num_frames])

            assert self.num_frames <= len(self.gt_annos), \
                'Number of frames is higher than length of annotations available \n'

    def download(self, path, split_file):
        return None

    def __getitem__(self, idx, **kwargs):

        with_label = kwargs.get('with_label', False)
        words = self.val_image_ids[idx].split(' ')
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
        #predictions = np.load('/user/a0393749/deepak_files/github/mmdetection3d-work/edgeai-mmdetection3d/bin_out.bin.npy',allow_pickle=True)
        dt_annos = self.bbox2result_kitti(predictions,self.class_names)
        acc = get_official_eval_result(self.gt_annos, dt_annos, self.class_names)
        # TODO: find this result from the complete result
        accuracy_ap_3d_moderate = float(acc[0].split('\n')[3].split(' ')[4].split(',')[0])
        ap_dict = {'accuracy_ap_3d_moderate%': accuracy_ap_3d_moderate, 'KITTI_official_eval_result':acc[0]}
        return ap_dict

    # https://github.com/open-mmlab/mmdetection3d/
    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            os.mkdir(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')

        for idx, pred in enumerate(net_outputs):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        return det_annos

    # https://github.com/open-mmlab/mmdetection3d/
    def convert_valid_bboxes(self, pred, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function

        # box predictions are in lidar refrence co-ordinate. Kitti ground truths are in kitti camera co0ordinates
        box_preds = pred[:, 2:8] # x y z l w h
        scores = pred[:, 1]
        labels = pred[:, 0]
        sample_idx = info['image']['image_idx']
        pred[:,8] = self.limit_period(pred[:,8], offset = 0.5, period = np.pi*2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        img_shape = info['image']['image_shape']
        #P2 = box_preds.tensor.new_tensor(P2)

        z = np.ones((len(box_preds),1))
        box_preds_with_ones = np.append(box_preds[:,0:3], z, axis=1)

        box_preds_camera = box_preds_with_ones @ ((rect @ Trv2c).transpose())[:,:3]

        box_size_camera = box_preds[:, 3:6].copy()
        #swap index 2 and 1, as height is length in camera view
        box_size_camera[:,2] = box_preds[:,4].copy()
        box_size_camera[:, 1] = box_preds[:, 5].copy()

        yaw_in_camera_view = -pred[:,8] - np.pi / 2
        yaw_in_camera_view = self.limit_period(yaw_in_camera_view, period=np.pi * 2)

        box_preds_camera = np.concatenate((box_preds_camera, box_size_camera, yaw_in_camera_view.reshape(-1, 1)), axis=1)
        box_corners = self.corners(box_preds_camera) # from camera co-ordinate

        box_corners_in_image = self.points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = np.min(box_corners_in_image, axis=1)
        maxxy = np.max(box_corners_in_image, axis=1)
        box_2d_preds = np.concatenate((minxy, maxxy), axis=1)
        # Post-processing
        # check box_preds_camera
        #image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < img_shape[1]) &
                          (box_2d_preds[:, 1] < img_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        #limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        limit_range = self.pcd_limit_range

        valid_pcd_inds = ((box_preds[:,:3] > limit_range[:3]) &
                          (box_preds[:,:3] < limit_range[3:]))
        valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :],
                box3d_camera=box_preds_camera[valid_inds],
                box3d_lidar=box_preds[valid_inds],
                scores=scores[valid_inds],
                label_preds=labels[valid_inds],
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

    # https://github.com/open-mmlab/mmdetection3d/
    def read_db_info(self, idx, velodyne = True, with_imageshape=True, calib=True, with_plane = False, label_info = True, extend_matrix = True):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        relative_path = False
        root_path =''
        if velodyne:
            pc_info['velodyne_path'] = os.path.join(self.kwargs['path'],self.kwargs['split'],self.kwargs['pts_prefix'], idx) + '.bin'
        image_info['image_path'] = os.path.join(self.kwargs['path'],self.kwargs['split'],'image_2', idx) + '.png'

        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                cv2.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = os.path.join(self.kwargs['path'],self.kwargs['split'],'label_2', idx) + '.txt'
            if relative_path:
                label_path = str(root_path / label_path)
            #annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = os.path.join(self.kwargs['path'], self.kwargs['split'],'calib', idx) + '.txt'
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = self._extend_matrix(P0)
                P1 = self._extend_matrix(P1)
                P2 = self._extend_matrix(P2)
                P3 = self._extend_matrix(P3)
            R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
            Tr_imu_to_velo = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = self._extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = self._extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info['calib'] = calib_info

        return info

    def _extend_matrix(self, mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat
    #https://github.com/open-mmlab/mmdetection3d/
    #https://github.com/open-mmlab/mmdetection3d/
    def points_cam2img(self, points_3d, proj_mat, with_depth=False):
        """Project points in camera coordinates to image coordinates.

        Args:
            points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
            proj_mat (torch.Tensor | np.ndarray):
                Transformation matrix between coordinates.
            with_depth (bool, optional): Whether to keep depth in the output.
                Defaults to False.

        Returns:
            (torch.Tensor | np.ndarray): Points in image coordinates,
                with shape [N, 2] if `with_depth=False`, else [N, 3].
        """
        points_shape = list(points_3d.shape)
        points_shape[-1] = 1

        assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
            f' matrix should be 2 instead of {len(proj_mat.shape)}.'
        d1, d2 = proj_mat.shape[:2]
        assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
            d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
            f' ({d1}*{d2}) is not supported.'
        if d1 == 3:
            proj_mat_expanded = torch.eye(
                4, device=proj_mat.device, dtype=proj_mat.dtype)
            proj_mat_expanded[:d1, :d2] = proj_mat
            proj_mat = proj_mat_expanded

        # previous implementation use new_zeros, new_one yields better results
        points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)

        point_2d = points_4 @ proj_mat.T
        point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

        if with_depth:
            point_2d_res = np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)

        return point_2d_res

    def boxes3d_lidar_to_kitti_camera(self, boxes3d_lidar, calib):
        """
        :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        :param calib:
        :return:
            boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        """
        boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
        xyz_lidar = boxes3d_lidar_copy[:, 0:3]
        l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
        r = boxes3d_lidar_copy[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = calib.lidar_to_rect(xyz_lidar)
        # xyz_cam[:, 1] += h.reshape(-1) / 2
        r = -r - np.pi / 2
        return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

    def limit_period(self, val, offset=0.5, period=np.pi):
        """Limit the value into a period for periodic function.

        Args:
            val (torch.Tensor | np.ndarray): The value to be converted.
            offset (float, optional): Offset to set the value range.
                Defaults to 0.5.
            period ([type], optional): Period of the value. Defaults to np.pi.

        Returns:
            (torch.Tensor | np.ndarray): Value in the range of
                [-offset * period, (1-offset) * period]
        """
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val

    def corners(self, tensor):
        """torch.Tensor: Coordinates of corners of all the boxes in
                         shape (N, 8, 3).

        Convert the boxes to  in clockwise order, in the form of
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)

        .. code-block:: none

                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v
                    down y
        """
        if np.size(tensor) == 0:
            return np.empty([0, 8, 3])

        dims = tensor[:,3:6]
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 1, 0.5]
        corners_norm = corners_norm - np.array([0.5, 1, 0.5])
        corners = np.expand_dims(dims, axis=1) * corners_norm.reshape([1, 8, 3])

        corners = self.rotation_3d_in_axis(
            corners, tensor[:, 6], axis=1)
        corners += np.expand_dims(tensor[:, :3], axis=1)
        return corners

    def rotation_3d_in_axis(self, points,
                            angles,
                            axis=0,
                            return_mat=False,
                            clockwise=False):
        """Rotate points by angles according to axis.

        Args:
            points (np.ndarray | torch.Tensor | list | tuple ):
                Points of shape (N, M, 3).
            angles (np.ndarray | torch.Tensor | list | tuple | float):
                Vector of angles in shape (N,)
            axis (int, optional): The axis to be rotated. Defaults to 0.
            return_mat: Whether or not return the rotation matrix (transposed).
                Defaults to False.
            clockwise: Whether the rotation is clockwise. Defaults to False.

        Raises:
            ValueError: when the axis is not in range [0, 1, 2], it will
                raise value error.

        Returns:
            (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
        """
        batch_free = len(points.shape) == 2
        if batch_free:
            points = points[None]

        if isinstance(angles, float) or len(angles.shape) == 0:
            angles = torch.full(points.shape[:1], angles)

        assert len(points.shape) == 3 and len(angles.shape) == 1 \
            and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
            f'angles: {points.shape}, {angles.shape}'

        assert points.shape[-1] in [2, 3], \
            f'Points size should be 2 or 3 instead of {points.shape[-1]}'

        rot_sin = np.sin(angles)
        rot_cos = np.cos(angles)
        ones = np.ones_like(rot_cos)
        zeros = np.zeros_like(rot_cos)

        if points.shape[-1] == 3:
            if axis == 1 or axis == -2:
                rot_mat_T = np.stack([
                    np.stack([rot_cos, zeros, -rot_sin]),
                    np.stack([zeros, ones, zeros]),
                    np.stack([rot_sin, zeros, rot_cos])
                ])
            elif axis == 2 or axis == -1:
                rot_mat_T = np.stack([
                    np.stack([rot_cos, rot_sin, zeros]),
                    np.stack([-rot_sin, rot_cos, zeros]),
                    np.stack([zeros, zeros, ones])
                ])
            elif axis == 0 or axis == -3:
                rot_mat_T = np.stack([
                    np.stack([ones, zeros, zeros]),
                    np.stack([zeros, rot_cos, rot_sin]),
                    np.stack([zeros, -rot_sin, rot_cos])
                ])
            else:
                raise ValueError(f'axis should in range '
                                 f'[-3, -2, -1, 0, 1, 2], got {axis}')
        else:
            rot_mat_T = np.stack([
                np.stack([rot_cos, rot_sin]),
                np.stack([-rot_sin, rot_cos])
            ])

        if clockwise:
            rot_mat_T = rot_mat_T.transpose(0, 1)

        if points.shape[0] == 0:
            points_new = points
        else:
            points_new = np.einsum('aij,jka->aik', points, rot_mat_T)

        if batch_free:
            points_new = points_new.squeeze(0)

        if return_mat:
            rot_mat_T = np.einsum('jka->ajk', rot_mat_T)
            if batch_free:
                rot_mat_T = rot_mat_T.squeeze(0)
            return points_new, rot_mat_T
        else:
            return points_new
