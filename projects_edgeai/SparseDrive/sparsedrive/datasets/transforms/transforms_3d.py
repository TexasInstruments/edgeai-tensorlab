import numpy as np
import torch
import mmcv

from mmcv.transforms import BaseTransform
from mmdet3d.structures.bbox_3d.utils import limit_period
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import to_tensor


@TRANSFORMS.register_module()
class CircleObjectRangeFilter(BaseTransform):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def transform(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = torch.sqrt(
            torch.sum(gt_bboxes_3d.tensor[:, :2] ** 2, axis=-1)
        )
        mask = torch.Tensor([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = torch.logical_or(
                mask,
                torch.logical_and(torch.from_numpy(gt_labels_3d) == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        #gt_labels_3d = gt_labels_3d[mask]
        gt_labels_3d = gt_labels_3d[mask.numpy()]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask.numpy()]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask.numpy()]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask.numpy()]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str

@TRANSFORMS.register_module()
class NuScenesSparse4DAdaptor(BaseTransform):
    def __init(self):
        pass

    def transform(self, input_dict):
        input_dict["projection_mat"] = to_tensor(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = to_tensor(input_dict["img_shape"])[:, :2][:, [1, 0]].to(torch.float32)

        input_dict["T_global_inv"] = to_tensor(np.linalg.inv(input_dict["lidar2global"]))
        input_dict["T_global"] = to_tensor(input_dict["lidar2global"])
        if "cam2img" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam2img"])
            )
            input_dict["focal"] = to_tensor(input_dict["cam_intrinsic"][..., 0, 0])
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"].tensor[:, 6] = limit_period(
                input_dict["gt_bboxes_3d"].tensor[:, 6], offset=0.5, period=2 * np.pi
            )
            #input_dict["gt_bboxes_3d"] = to_tensor(input_dict["gt_bboxes_3d"]).float()
        #if "gt_labels_3d" in input_dict:
        #    input_dict["gt_labels_3d"] = to_tensor(input_dict["gt_labels_3d"]).long()

        #imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        #imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        #input_dict["img"] = DC(to_tensor(imgs), stack=True)

        if "ann_info" in input_dict:
            input_dict["gt_ego_fut_cmd"] = input_dict['ann_info']['gt_ego_fut_cmd']

        # To REVISTI
        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            #input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 
            input_dict[key] = to_tensor(input_dict[key])

        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ]:
            if key not in input_dict:
                continue
            #input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
            input_dict[key] = to_tensor(input_dict[key])

        return input_dict


@TRANSFORMS.register_module()
class MultiScaleDepthMapGenerator(BaseTransform):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def transform(self, input_dict):
        points = input_dict["points"].tensor[..., :3, None].numpy()
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [to_tensor(np.stack(x)) for x in gt_depth]
        return input_dict


@TRANSFORMS.register_module()
class BBoxRotation(BaseTransform):
    def __init__(self, data_aug_conf=None, training=True):
        assert data_aug_conf is not None and training is True
        self.data_aug_conf = data_aug_conf
        self.training = training

    def transform(self, results):
        if self.training:
            angle = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            angle = 0.0

        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            #results["gt_bboxes_3d"] = self.box_rotate(
            #    results["gt_bboxes_3d"], angle
            #)
            results["gt_bboxes_3d"].rotate(angle)
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = torch.from_numpy(np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        ))
        bbox_3d.tensor[:, :3] = bbox_3d.tensor[:, :3] @ rot_mat_T
        bbox_3d.tensor[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d.tensor[:, 7:].shape[-1]
            bbox_3d.tensor[:, 7:] = bbox_3d.tensor[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d



@TRANSFORMS.register_module()
class InstanceNameFilter(BaseTransform):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def transform(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=bool
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(BaseTransform):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        bgr_to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, bgr_to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.bgr_to_rgb = bgr_to_rgb

    def transform(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.bgr_to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, bgr_to_rgb=self.bgr_to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, bgr_to_rgb={self.bgr_to_rgb})'
        return repr_str

