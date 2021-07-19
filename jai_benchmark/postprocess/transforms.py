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

import os
import copy
import numpy as np
import cv2
from PIL import ImageDraw
from munkres import Munkres
from numpy.lib.stride_tricks import as_strided
import math


##############################################################################
class IndexArray():
    def __init__(self, index=0):
        self.index = index

    def __call__(self, input, info_dict):
        return input[self.index], info_dict


class ArgMax():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor, info_dict):
        if self.axis is None:
            axis = 1 if tensor.ndim == 4 else 0
        else:
            axis = self.axis
        #
        if tensor.shape[axis] > 1:
            tensor = tensor.argmax(axis=axis)
            tensor = tensor[0]
        #
        return tensor, info_dict


class Concat():
    def __init__(self, axis=-1, start_index=0, end_index=-1):
        self.axis = axis
        self.start_index = start_index
        self.end_index = end_index

    def __call__(self, tensor_list, info_dict):
        if isinstance(tensor_list, (list,tuple)):
            max_dim = 0
            for t_idx, t in enumerate(tensor_list):
                max_dim = max(max_dim, t.ndim)
            #
            for t_idx, t in enumerate(tensor_list):
                if t.ndim < max_dim:
                    tensor_list[t_idx] = t[...,np.newaxis]
                #
            #
            tensor = np.concatenate(tensor_list[self.start_index:self.end_index], axis=self.axis)
        else:
            tensor = tensor_list
        #
        return tensor, info_dict


class ShuffleList():
    def __init__(self, indices=None):
        self.indices = indices

    def __call__(self, tensor_list, info_dict):
        if self.indices is not None:
            tensor_list_out = []
            for ind in self.indices:
                tensor_list_out.append(tensor_list[ind])
            #
        else:
            tensor_list_out = tensor_list
        #
        return tensor_list_out, info_dict


class ReshapeList():
    def __init__(self, reshape_list=None):
        self.reshape_list = reshape_list

    def __call__(self, tensor_list, info_dict):
        if self.reshape_list is not None:
            tensor_list_out = []
            for t_orig, t_shape in zip(tensor_list, self.reshape_list):
                tensor_list_out.append(t_orig.reshape(t_shape))
            #
        else:
            tensor_list_out = tensor_list
        #
        return tensor_list_out, info_dict


class IgnoreDetectionElement():
    def __init__(self, indice=None):
        self.indice = indice

    def __call__(self, tensor, info_dict):
        if self.indice is not None:
            tensor_out = np.concatenate((tensor[...,:self.indice], tensor[...,self.indice+1:]) ,-1)
        #
        else:
            tensor_out = tensor
        #
        return tensor_out, info_dict


##############################################################################
class SegmentationImageResize():
    def __call__(self, label, info_dict):
        image_shape = info_dict['data_shape']
        if label.dtype in (np.int32, np.int64):
            label = label.astype(np.float32)
        #
        label = cv2.resize(label, dsize=(image_shape[1],image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return label, info_dict


class SegmentationImagetoBytes():
    '''
    Convert Segmentation image to bytes (uint8) to save space
    '''
    def __call__(self, label, info_dict):
        label = label.astype(np.uint8)
        return label, info_dict


class SegmentationImageSave():
    def __init__(self):
        self.colors = [(r,g,b) for r in range(0,256,32) for g in range(0,256,32) for b in range(0,256,32)]

    def __call__(self, tensor, info_dict):
        data_path = info_dict['data_path']
        # img_data = info_dict['data']
        image_name = os.path.split(data_path)[-1].split('.')[0] + '.png'
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        # TODO: convert label to color here
        if isinstance(tensor, np.ndarray):
            # convert image to BGR
            tensor = tensor[:,:,::-1] if tensor.ndim > 2 else tensor
            cv2.imwrite(save_path, tensor)
        else:
            # add fill code here
            tensor.save(save_path)
        #
        return tensor, info_dict


##############################################################################
class DetectionResizeOnlyNormalized():
    def __call__(self, bbox, info_dict):
        img_data = info_dict['data']
        assert isinstance(img_data, np.ndarray), 'only supports np array for now'
        data_shape = info_dict['data_shape']
        data_height, data_width, _ = data_shape
        # avoid accidental overflow
        bbox = bbox.clip(-1e6, 1e6)
        # scale the detections from normalized shape (0-1) to data shape
        bbox[...,0] = (bbox[...,0]*data_width).clip(0, data_width)
        bbox[...,1] = (bbox[...,1]*data_height).clip(0, data_height)
        bbox[...,2] = (bbox[...,2]*data_width).clip(0, data_width)
        bbox[...,3] = (bbox[...,3]*data_height).clip(0, data_height)
        return bbox, info_dict


class DetectionResizePad():
    def __init__(self, resize_with_pad=False, normalized_detections=True):
        self.resize_with_pad = resize_with_pad
        self.normalized_detections = normalized_detections

    def __call__(self, bbox, info_dict):
        img_data = info_dict['data']
        assert isinstance(img_data, np.ndarray), 'only supports np array for now'
        # avoid accidental overflow
        bbox = bbox.clip(-1e6, 1e6)
        # img size without pad
        data_shape = info_dict['data_shape']
        data_height, data_width, _ = data_shape
        resize_shape = info_dict['resize_shape']
        resize_height, resize_width, _ = resize_shape
        if self.resize_with_pad:
            # account for padding
            border = info_dict['resize_border']
            left, top, right, bottom = border
            bbox[...,0] -= left
            bbox[...,1] -= top
            bbox[...,2] -= left
            bbox[...,3] -= top
            resize_height, resize_width = (resize_height - top - bottom), (resize_width - left - right)
        #
        # scale the detections from the input shape to data shape
        sh = data_height / (1.0 if self.normalized_detections else resize_height)
        sw = data_width / (1.0 if self.normalized_detections else resize_width)
        bbox[...,0] = (bbox[...,0] * sw).clip(0, data_width)
        bbox[...,1] = (bbox[...,1] * sh).clip(0, data_height)
        bbox[...,2] = (bbox[...,2] * sw).clip(0, data_width)
        bbox[...,3] = (bbox[...,3] * sh).clip(0, data_height)
        return bbox, info_dict


class DetectionFilter():
    def __init__(self, detection_thr, detection_max=None):
        self.detection_thr = detection_thr
        self.detection_max = detection_max

    def __call__(self, bbox, info_dict):
        if self.detection_thr is not None:
            bbox_score = bbox[:,5]
            bbox_selected = (bbox_score >= self.detection_thr)
            bbox = bbox[bbox_selected,...]
        #
        if self.detection_max is not None and bbox.shape[0] > self.detection_max:
            bbox = sorted(bbox, key=lambda b:b[5])
            bbox = np.stack(bbox, axis=0)
            bbox = bbox[range(self.detection_max),...]
        #
        return bbox, info_dict


class DetectionFormatting():
    def __init__(self, dst_indices, src_indices):
        self.src_indices = src_indices
        self.dst_indices = dst_indices

    def __call__(self, bbox, info_dict):
        bbox_copy = copy.deepcopy(bbox)
        bbox_copy[...,self.dst_indices] = bbox[...,self.src_indices]
        return bbox_copy, info_dict


class DetectionXYXY2YXYX(DetectionFormatting):
    def __init__(self, dst_indices=(0,1,2,3), src_indices=(1,0,3,2)):
        super().__init__(dst_indices, src_indices)


class DetectionYXYX2XYXY(DetectionFormatting):
    def __init__(self, dst_indices=(0,1,2,3), src_indices=(1,0,3,2)):
        super().__init__(dst_indices, src_indices)


class DetectionYXHW2XYWH(DetectionFormatting):
    def __init__(self, dst_indices=(0,1,2,3), src_indices=(1,0,3,2)):
        super().__init__(dst_indices, src_indices)


class DetectionXYXY2XYWH():
    def __call__(self, bbox, info_dict):
        w = bbox[...,2] - bbox[...,0]
        h = bbox[...,3] - bbox[...,1]
        bbox[...,2] = w
        bbox[...,3] = h
        return bbox, info_dict


class DetectionXYWH2XYXY():
    def __call__(self, bbox, info_dict):
        x2 = bbox[...,0] + bbox[...,2]
        y2 = bbox[...,1] + bbox[...,3]
        bbox[...,2] = x2
        bbox[...,3] = y2
        return bbox, info_dict


class DetectionBoxSL2BoxLS(DetectionFormatting):
    def __init__(self, dst_indices=(4,5), src_indices=(5,4)):
        super().__init__(dst_indices, src_indices)


class DetectionImageSave():
    def __init__(self):
        self.colors = [(r,g,b) for r in range(0,256,32) for g in range(0,256,32) for b in range(0,256,32)]
        self.thickness = 2

    def __call__(self, bbox, info_dict):
        data_path = info_dict['data_path']
        img_data = info_dict['data']
        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        img_data = copy.deepcopy(img_data)
        if isinstance(img_data, np.ndarray):
            for bbox_one in bbox:
                label = int(bbox_one[4])
                label_color = self.colors[label % len(self.colors)]
                pt1 = (int(bbox_one[0]),int(bbox_one[1]))
                pt2 = (int(bbox_one[2]),int(bbox_one[3]))
                cv2.rectangle(img_data, pt1, pt2, color=label_color, thickness=self.thickness)
            #
            cv2.imwrite(save_path, img_data[:,:,::-1])
        else:
            img_rect = ImageDraw.Draw(img_data)
            for bbox_one in bbox:
                label = int(bbox_one[4])
                label_color = self.colors[label % len(self.colors)]
                rect = (int(bbox_one[0]),int(bbox_one[1]),int(bbox_one[2]),int(bbox_one[3]))
                img_rect.rectangle(rect, outline=label_color, width=self.thickness)
            #
            img_data.save(save_path)
        #
        return bbox, info_dict



##############################################################################
class NPTensorToImage(object):
    def __init__(self, data_layout='NCHW'):
        self.data_layout = data_layout

    def __call__(self, tensor, info_dict):
        assert isinstance(tensor, np.ndarray), 'input tensor must be an array'
        if tensor.ndim >= 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        #
        if tensor.ndim==2:
            if self.data_layout=='NHWC':
                tensor = tensor[..., np.newaxis]
            else:
                tensor = tensor[np.newaxis, ...]
        assert tensor.ndim == 3, 'could not convert to image'
        tensor = np.transpose(tensor, (1,2,0)) if self.data_layout == 'NCHW' else tensor
        assert tensor.shape[2] in (1,3), 'invalid number of channels'
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout})'


##############################################################################
     
class HumanPoseHeatmapParser:
    def __init__(self, use_udp=True):
        self.num_joints = 17
        self.max_num_people = 30
        self.detection_threshold = 0.1
        self.tag_threshold = 1
        self.use_detection_val  = True
        self.ignore_too_much = False
        self.joint_order = [0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16]
        self.tag_per_joint = True
        self.nms_kernel = 5
        self.nms_padding = 2
        self.use_udp = use_udp
        self.with_heatmaps = [True] 
        self.with_ae = [True]
        self.flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        self.project2image = not(use_udp)
        self.align_corners = use_udp
        self.adjust_keypoints = True
        self.refine_keypoints = True

    def maxpool2d(self,A):
        """Apply the max pool operation on array A with self.nms_kernel as kernel size,
        self.nms_padding as padding size and can specify the stride in function itself

        Args:
            A(np.ndarray[WxH]): heatmap array

        Returns:
            np.ndarray[WxH] : max pooled heatmap array
        """
        A = np.pad(A, self.nms_padding, mode='constant')
        stride = 1
        output_shape = ((A.shape[0] - self.nms_kernel)//stride + 1,
                        (A.shape[1] - self.nms_kernel)//stride + 1)
        kernel_size = (self.nms_kernel, self.nms_kernel)
        A_w = as_strided(A, shape = output_shape + kernel_size, 
                            strides = (stride*A.strides[0],
                                    stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)
        return A_w.max(axis=(1,2)).reshape(output_shape)

    def _py_max_match(self, scores):
        """Apply munkres algorithm to get the best match.

        Args:
            scores(np.ndarray): cost matrix.

        Returns:
            np.ndarray: best match.
        """
        m = Munkres()
        tmp = m.compute(scores)
        tmp = np.array(tmp).astype(int)
        return tmp

    def _match_by_tag(self, inp):
        """Match joints by tags. Use Munkres algorithm to calculate the best match
        for keypoints grouping.

        Note:
            number of keypoints: K
            max number of people in an image: M (M=30 by default)
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            inp(tuple):
                tag_k (np.ndarray[KxMxL]): tag corresponding to the
                    top k values of feature map per keypoint.
                loc_k (np.ndarray[KxMx2]): top k locations of the
                    feature maps for keypoint.
                val_k (np.ndarray[KxM]): top k value of the
                    feature maps per keypoint.

        Returns:
            np.ndarray: result of pose groups.
        """

        tag_k, loc_k, val_k = inp

        default_ = np.zeros((self.num_joints, 3 + tag_k.shape[2]),
                            dtype=np.float32)

        joint_dict = {}
        tag_dict = {}
        for i in range(self.num_joints):
            idx = self.joint_order[i]

            tags = tag_k[idx]
            joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if i == 0 or len(joint_dict) == 0:
                for tag, joint in zip(tags, joints):
                    key = tag[0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                    tag_dict[key] = [tag]
            else:
                grouped_keys = list(joint_dict.keys())[:self.max_num_people]
                grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

                if (self.ignore_too_much
                        and len(grouped_keys) == self.max_num_people):
                    continue

                diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
                diff_normed = np.linalg.norm(diff, ord=2, axis=2)
                diff_saved = np.copy(diff_normed)

                if self.use_detection_val:
                    diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

                num_added = diff.shape[0]
                num_grouped = diff.shape[1]

                if num_added > num_grouped:
                    diff_normed = np.concatenate(
                        (diff_normed,
                        np.zeros((num_added, num_added - num_grouped),
                                dtype=np.float32) + 1e10),
                        axis=1)

                pairs = self._py_max_match(diff_normed)
                for row, col in pairs:
                    if (row < num_added and col < num_grouped
                            and diff_saved[row][col] < self.tag_threshold):
                        key = grouped_keys[col]
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key].append(tags[row])
                    else:
                        key = tags[row][0]
                        joint_dict.setdefault(key, np.copy(default_))[idx] = \
                            joints[row]
                        tag_dict[key] = [tags[row]]

        ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
        return ans

    def gather(self, a, dim, index):
        """Get the corresponding values from "a" matrix along the "dim" dimension according to "index" array 
        """
        expanded_index = [index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
        return a[tuple(expanded_index)]

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.
        """
        heatmaps_new = copy.deepcopy(heatmaps)
        for i,heatmap in enumerate(heatmaps[0]):
            maxm = self.maxpool2d(heatmap)
            maxm = np.equal(maxm,heatmap)
            heatmap = heatmap * maxm
            heatmaps_new[0][i] = heatmap
        return heatmaps_new

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list of matched keypoints of size N with each element as np.ndarray[PxKx4]
            where P : number of detected people in a image
        """

        def _match(x):
            return self._match_by_tag(x)

        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in an image.

        Note:
            batch size: N ==1
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (np.ndarray[NxKxHxW])
            tags (np.ndarray[NxKxHxWxL])

        Return:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """
        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.shape
        heatmaps = np.reshape(heatmaps,[N,K,-1])
        
        ind = np.zeros((N,K,self.max_num_people),int)
        val_k = np.zeros((N,K,self.max_num_people))
        for i,heatmap in enumerate(heatmaps[0]):
            ind[0][i] = heatmap.argsort()[-self.max_num_people:][::-1]
            val_k[0][i] = heatmap[ind[0][i]]
            
        tags = np.reshape(tags,(tags.shape[0], tags.shape[1], W * H, -1))
        tag_k = np.concatenate([np.expand_dims(self.gather(tags[...,i],2,ind),axis=3) for i in range(tags.shape[3])],axis=3)

        x = ind % W
        y = ind // W

        ind_k = np.concatenate((np.expand_dims(x,axis=3),np.expand_dims(y,axis=3)), axis=3)

        ans = {
            'tag_k': tag_k,
            'loc_k': ind_k,
            'val_k': val_k
        }
        
        return ans

    def adjust(self, ans, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            ans (list(np.ndarray)): Keypoint predictions.
            heatmaps (np.ndarray[NxKxHxW]): Heatmaps.
        """
        _, _, H, W = heatmaps.shape
        for batch_id, people in enumerate(ans):
            for people_id, people_i in enumerate(people):
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        x, y = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = heatmaps[batch_id][joint_id]
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1),
                                                             xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy,
                                                             max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id,
                                      0:2] = (x + 0.5, y + 0.5)
        return ans

    def refine(self, heatmap, tag, keypoints):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).

        Returns:
            np.ndarray: The refined keypoints.
        """

        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        tags = []
        for i in range(K):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(int)
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            distance_tag = (((_tag -
                              prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            norm_heatmap = _heatmap - np.round(distance_tag)

            # find maximum position
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            val = _heatmap[y, x]
            if not self.use_udp:
                # offset by 0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(K):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :3] = ans[i, :3]

        return keypoints

    def post_dark_udp(self, coords, batch_heatmaps, kernel=3):
        """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
        Devil is in the Details: Delving into Unbiased Data Processing for Human
        Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).

        Note:
            batch size: B
            num keypoints: K
            num persons: N
            hight of heatmaps: H
            width of heatmaps: W
            B=1 for bottom_up paradigm where all persons share the same heatmap.
            B=N for top_down paradigm where each person has its own heatmaps.

        Args:
            coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
            batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
            kernel (int): Gaussian kernel size (K) for modulation.

        Returns:
            res (np.ndarray[N, K, 2]): Refined coordinates.
        """
        batch_heatmaps = copy.deepcopy(batch_heatmaps)
        B, K, H, W = batch_heatmaps.shape
        N = coords.shape[0]
        assert (B == 1 or B == N)
        for heatmaps in batch_heatmaps:
            for heatmap in heatmaps:
                cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
        np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
        np.log(batch_heatmaps, batch_heatmaps)
        batch_heatmaps = np.transpose(batch_heatmaps,
                                    (2, 3, 0, 1)).reshape(H, W, -1)
        batch_heatmaps_pad = cv2.copyMakeBorder(
            batch_heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
        batch_heatmaps_pad = np.transpose(
            batch_heatmaps_pad.reshape(H + 2, W + 2, B, K),
            (2, 3, 0, 1)).flatten()

        index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
        index = index.astype(np.int).reshape(-1, 1)
        i_ = batch_heatmaps_pad[index]
        ix1 = batch_heatmaps_pad[index + 1]
        iy1 = batch_heatmaps_pad[index + W + 2]
        ix1y1 = batch_heatmaps_pad[index + W + 3]
        ix1_y1_ = batch_heatmaps_pad[index - W - 3]
        ix1_ = batch_heatmaps_pad[index - 1]
        iy1_ = batch_heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(N, K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
        return coords

    def parse(self, heatmaps, tags):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (np.ndarray[NxKxHxW]): model output heatmaps.
            tags (np.ndarray[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - ans (list(np.ndarray)): Pose results.
            - scores (list): Score of people.
        """
        ans = self.match(**self.top_k(heatmaps, tags))

        if self.adjust_keypoints:
            if self.use_udp:
                for i in range(len(ans)):
                    if ans[i].shape[0] > 0:
                        ans[i][..., :2] = self.post_dark_udp(
                            ans[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                ans = self.adjust(ans, heatmaps)

        scores = [i[:, 2].mean() for i in ans[0]]

        if self.refine_keypoints:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                heatmap_numpy = heatmaps[0]
                tag_numpy = tags[0]
                if not self.tag_per_joint:
                    tag_numpy = np.tile(tag_numpy,
                                        (self.num_joints, 1, 1, 1))
                ans[i] = self.refine(
                    heatmap_numpy, tag_numpy, ans[i])
            ans = [ans]

        return ans, scores

    def PoseMultiStage(self, outputs, outputs_flip, base_size):
        """ Outputs is divided into heatmaps and tags according to num_joints label.
        If flip test is set to true then outputs_flip have output of inference of flipped image else it is None.
        If we want to project the heatmaps to image, the project_to_image flag is used and the heatmaps are projected 
        to the specified base_size. The flipped outputs are also projected if flip test is used.
        If flip test is true, then the heatmaps from flipped outputs and original outputs are averaged and the tags concatenated.

        Note:
        N - Number of images for inference == 1
        K - Number of keypoints (specified using self.num_joints)
        WxH - Output Shape from the network inference

        Inputs:
        outputs : np.ndarray[Nx2*KxWxH] - outputs from network inference
        outputs_flip : np.ndarray[Nx2*KxWxH] if flip test else None - outputs from network inference using flipped image
        base_size: (W_input,H_input) - size of input to the inference network

        Outputs:
        aggregated_heatmaps : np.ndarray
        tags : np.ndarray
        """

        heatmaps = [outputs[0][:, :self.num_joints]]
        tags = [outputs[0][:, self.num_joints:]]
        
        aggregated_heatmaps = None
        tags_list = []

        flip_test = outputs_flip is not None

        if flip_test and self.flip_index:
            # perform flip testing      
            outputs_flip[0] = np.flip(outputs_flip[0], axis=3)
            heatmaps_avg = outputs_flip[0][:, :self.num_joints][:, self.flip_index, :, :]
            tags.append(outputs_flip[0][:, self.num_joints:])
            if self.tag_per_joint:
                tags[-1] = tags[-1][:, self.flip_index, :, :]
            heatmaps.append(heatmaps_avg)

        if self.project2image and base_size:
            # project it to the actual image size
            dim = (base_size[1], base_size[0])

            final_heatmaps =[]
            final_tags = []
            
            new_heatmaps = np.empty((0,dim[0],dim[1]),int)
            for hms in heatmaps[0][0]:
                new_hms = cv2.resize(
                    hms,
                    dim,
                    interpolation=cv2.INTER_LINEAR)
                new_heatmaps = np.append(new_heatmaps,[new_hms],axis=0)
            
            final_heatmaps.append(np.expand_dims(new_heatmaps,0))

            if flip_test:
                new_heatmaps_flipped = np.empty((0,dim[0],dim[1]),int)
                for hms in heatmaps[1][0]:
                    new_hms = cv2.resize(
                        hms,
                        dim,
                        interpolation=cv2.INTER_LINEAR)
                    new_heatmaps_flipped = np.append(new_heatmaps_flipped,[new_hms],axis=0)
                    
                final_heatmaps.append(np.expand_dims(new_heatmaps_flipped,0))

            new_tags = np.empty((0,dim[0],dim[1]),int)
            for tms in tags[0][0]:
                new_tms =  cv2.resize(
                    tms,
                    dim,
                    interpolation=cv2.INTER_LINEAR)
                new_tags = np.append(new_tags,[new_tms],axis=0)
                
            final_tags.append(np.expand_dims(new_tags,0))

            if flip_test:
                new_tags_flipped = np.empty((0,dim[0],dim[1]),int)
                for tms in tags[1][0]:
                    new_tms = cv2.resize(
                        tms,
                        dim,
                        interpolation=cv2.INTER_LINEAR)
                    new_tags_flipped = np.append(new_tags_flipped,[new_tms],axis=0)
                    
                final_tags.append(np.expand_dims(new_tags_flipped,0))
        
        else:
            final_tags = tags
            final_heatmaps = heatmaps
                
        for tms in final_tags:
            tags_list.append(np.expand_dims(tms,axis=4))
            
        aggregated_heatmaps = (final_heatmaps[0] + 
                    final_heatmaps[1]) / 2.0 if flip_test else final_heatmaps[0] 
        
        tags = np.concatenate(tags_list,axis=4)
            
        return aggregated_heatmaps, tags

    def __call__(self, outputs, info_dict):
        outputs_flip = info_dict['outputs_flip']
        base_size = (info_dict['resize_shape'][0],info_dict['resize_shape'][1])

        heatmaps, tags = self.PoseMultiStage(outputs,outputs_flip,base_size)
        info_dict['heatmaps_shape'] = heatmaps.shape
        info_dict['tags_shape'] = tags.shape

        ans, scores = self.parse(heatmaps, tags)
        info_dict['scores'] = scores

        return ans, info_dict
    

class KeypointsProject2Image:
    def __init__(self, use_udp=True):
        self.use_udp = use_udp

    def get_warp_matrix(self, theta, size_input, size_dst, size_target):
        """Calculate the transformation matrix under the constraint of unbiased.
        Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
        Data Processing for Human Pose Estimation (CVPR 2020).

        Args:
            theta (float): Rotation angle in degrees.
            size_input (np.ndarray): Size of input image [w, h].
            size_dst (np.ndarray): Size of output image [w, h].
            size_target (np.ndarray): Size of ROI in input plane [w, h].

        Returns:
            matrix (np.ndarray): A matrix for transformation.
        """
        theta = np.deg2rad(theta)
        matrix = np.zeros((2, 3), dtype=np.float32)
        scale_x = size_dst[0] / size_target[0]
        scale_y = size_dst[1] / size_target[1]
        matrix[0, 0] = math.cos(theta) * scale_x
        matrix[0, 1] = -math.sin(theta) * scale_x
        matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                                0.5 * size_input[1] * math.sin(theta) +
                                0.5 * size_target[0])
        matrix[1, 0] = math.sin(theta) * scale_y
        matrix[1, 1] = math.cos(theta) * scale_y
        matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                                0.5 * size_input[1] * math.cos(theta) +
                                0.5 * size_target[1])
        return matrix

    def warp_affine_joints(self, joints, mat):
        """Apply affine transformation defined by the transform matrix on the
        joints.

        Args:
            joints (np.ndarray[..., 2]): Origin coordinate of joints.
            mat (np.ndarray[3, 2]): The affine matrix.

        Returns:
            matrix (np.ndarray[..., 2]): Result coordinate of joints.
        """
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(
            np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
            mat.T).reshape(shape)

    def transform_preds(self, coords, final_size, target_img_size):
        """Get final keypoint predictions from heatmaps and apply scaling and
        translation to map them back to the orignal image.

        Note:
            num_keypoints: K

        Args:
            coords (np.ndarray[K, ndims]):

                * If ndims=2, corrds are predicted keypoint location.
                * If ndims=4, corrds are composed of (x, y, scores, tags)
                * If ndims=5, corrds are composed of (x, y, scores, tags,
                flipped_tags)

            final_size (int): Image size of the input of the inference network 
            target_img_size (np.ndarray[2]): Final expected size of image (orignal image size before preprocessing)

        Returns:
            np.ndarray: Predicted coordinates in the images.
        """      

        if target_img_size[1]<target_img_size[0]:
            scale_it = final_size/target_img_size[0]
            value_it = (final_size - scale_it*target_img_size[1])/2
            coords[:,0] = coords[:,0]/scale_it
            coords[:,1] = (coords[:,1]-value_it)/scale_it
        else:
            scale_it = final_size/target_img_size[1]
            value_it = (final_size - scale_it*target_img_size[0])/2
            coords[:,1] = coords[:,1]/scale_it
            coords[:,0] = (coords[:,0]-value_it)/scale_it
        return coords

    def get_group_preds(self, grouped_joints, final_size, target_img_size, scale, heatmap_size):
        """Transform the grouped joints back to the image.

        Args:
            grouped_joints (list): Grouped person joints.
            final_size (int): Image size of the input of the inference network 
            target_img_size (np.ndarray[2]): Final expected size of image (orignal image size before preprocessing)
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.

        Returns:
            list: List of the pose result for each person.
        """
        if self.use_udp:
            if grouped_joints[0].shape[0] > 0:
                heatmap_size_t = np.array(heatmap_size, dtype=np.float32) - 1.0
                trans = self.get_warp_matrix(
                    theta=0,
                    size_input=heatmap_size_t,
                    size_dst=scale,
                    size_target=heatmap_size_t)
                grouped_joints[0][..., :2] = \
                    self.warp_affine_joints(grouped_joints[0][..., :2], trans)

        results = []
        for person in grouped_joints[0]:
            joints = self.transform_preds(person,final_size,target_img_size)
            results.append(joints)
        return results

    def __call__(self,  grouped_joints, info_dict):
        final_size = info_dict['resize_shape'][0]
        scale = np.array([final_size-1,final_size-1]) if self.use_udp \
             else np.array([final_size/200,final_size/200])
        heatmap_size = [info_dict['heatmaps_shape'][3],info_dict['heatmaps_shape'][2]]
        target_img_size = [info_dict['data_shape'][1],info_dict['data_shape'][0]]

        preds = self.get_group_preds(grouped_joints, final_size, target_img_size, scale, heatmap_size)

        image_paths = []
        image_paths.append(info_dict['data_path'])

        output_heatmap = None

        result = {}
        result['preds'] = preds
        result['scores'] = info_dict['scores']
        result['image_paths'] = image_paths
        result['output_heatmap'] = output_heatmap

        return result, info_dict
    

class HumanPoseImageSave:
    def __init__(self):
        self.pose_nms_thr = 0.9
        self.kpt_score_thr = 0.3
        self.palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])
        self.radius = 4
        self.thickness = 1
        self.font_scale = 0.5
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.pose_limb_color = self.palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        self.pose_kpt_color = self.palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
        ]]
        self.show_keypoint_weight = False

    def oks_iou(self, g, d, a_g, a_d, sigmas=None, vis_thr=None):
        """Calculate oks ious.

        Args:
            g: Ground truth keypoints.
            d: Detected keypoints.
            a_g: Area of the ground truth object.
            a_d: Area of the detected object.
            sigmas: standard deviation of keypoint labelling.
            vis_thr: threshold of the keypoint visibility.

        Returns:
            list: The oks ious.
        """
        if sigmas is None:
            sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                .87, .87, .89, .89
            ]) / 10.0
        vars = (sigmas * 2)**2
        xg = g[0::3]
        yg = g[1::3]
        vg = g[2::3]
        ious = np.zeros(len(d), dtype=np.float32)
        for n_d in range(0, len(d)):
            xd = d[n_d, 0::3]
            yd = d[n_d, 1::3]
            vd = d[n_d, 2::3]
            dx = xd - xg
            dy = yd - yg
            e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
            if vis_thr is not None:
                ind = list(vg > vis_thr) and list(vd > vis_thr)
                e = e[ind]
            ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
        return ious

    def oks_nms(self, kpts_db, thr, sigmas=None, vis_thr=None):
        """OKS NMS implementations.

        Args:
            kpts_db: keypoints.
            thr: Retain overlap < thr.
            sigmas: standard deviation of keypoint labelling.
            vis_thr: threshold of the keypoint visibility.

        Returns:
            np.ndarray: indexes to keep.
        """
        if len(kpts_db) == 0:
            return []

        scores = np.array([k['score'] for k in kpts_db])
        kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
        areas = np.array([k['area'] for k in kpts_db])

        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            oks_ovr = self.oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                            sigmas, vis_thr)

            inds = np.where(oks_ovr <= thr)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep

    def draw_and_save(self, img, result):
        img_h, img_w, _ = img.shape
        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        for _, kpts in enumerate(pose_result):
            # draw each point on image
            if self.pose_kpt_color is not None:
                assert len(self.pose_kpt_color) == len(kpts)
                for kid, kpt in enumerate(kpts):
                    x_coord, y_coord, kpt_score = int(kpt[0]), int(
                        kpt[1]), kpt[2]
                    if kpt_score > self.kpt_score_thr:
                        if self.show_keypoint_weight:
                            img_copy = img.copy()
                            r, g, b = self.pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                    self.radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                        else:
                            r, g, b = self.pose_kpt_color[kid]
                            cv2.circle(img, (int(x_coord), int(y_coord)),
                                    self.radius, (int(r), int(g), int(b)), -1)

            # draw limbs
            if self.skeleton is not None and self.pose_limb_color is not None:
                assert len(self.pose_limb_color) == len(self.skeleton)
                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                    pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                    if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                            and pos1[1] < img_h and pos2[0] > 0
                            and pos2[0] < img_w and pos2[1] > 0
                            and pos2[1] < img_h
                            and kpts[sk[0] - 1, 2] > self.kpt_score_thr
                            and kpts[sk[1] - 1, 2] > self.kpt_score_thr):
                        r, g, b = self.pose_limb_color[sk_id]
                        if self.show_keypoint_weight:
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)
                            cv2.fillConvexPoly(img_copy, polygon,
                                            (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                    (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                        else:
                            cv2.line(
                                img,
                                pos1,
                                pos2, (int(r), int(g), int(b)),
                                thickness=self.thickness)     
        return img

    def __call__(self, result, info_dict):
        data_path = info_dict['data_path']
        img_data = info_dict['data']
        image_name = os.path.split(data_path)[-1]
        run_dir = info_dict['run_dir']
        save_dir = os.path.join(run_dir, 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        pose_results = []
        for idx, pred in enumerate(result['preds']):
            area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                np.max(pred[:, 1]) - np.min(pred[:, 1]))
            pose_results.append({
                'keypoints': pred[:, :3],
                'score': result['scores'][idx],
                'area': area,
            })

        keep = self.oks_nms(pose_results, self.pose_nms_thr, sigmas=None)
        pose_results = [pose_results[_keep] for _keep in keep]

        img_data = copy.deepcopy(img_data[:,:,::-1])
        if isinstance(img_data, np.ndarray):
            img = self.draw_and_save(img_data, pose_results)
            cv2.imwrite(save_path, img)
        else:
            assert False, f'PIL image type isnt supported because PIL process dont pad right now' #TODO
        #
        return result, info_dict
