#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch

from yolox.exp import Exp as MyExp
import torch.distributed as dist
import torch.nn as nn


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "relu"

        # ---------------- model config ---------------- #
        self.num_classes = 15

        # ---------------- dataloader config ---------------- #
        self.input_size = (480, 640)  # (height, width)

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.0
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.9, 1.1)
        self.mixup_scale = (1.0, 1.0)
        self.shear = 0.0
        self.perspective = 0.0
        self.enable_mixup = False
        self.shape_loss = False
        # --------------  training config --------------------- #
        self.max_epoch = 300
        self.eval_interval = 10
        # -----------------  testing config ------------------ #
        self.test_size = (480, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.001
        self.data_set = "ycbv" # "lmo"
        self.object_pose  = True
        self.visualize = True
        self.od_weights = None

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXObjectPoseHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, conv_focus=True, split_max_pool_kernel=True)
            head = YOLOXObjectPoseHead(self.num_classes, self.width, in_channels=in_channels, dataset=self.data_set, shape_loss=self.shape_loss, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        if self.od_weights is not None:    
            state_dict_object_pose = self.model.state_dict()
            state_dict_object_detect = torch.load(self.od_weights, map_location=self.device)

            for key in state_dict_object_detect.keys():
                state_dict_object_pose[key] = state_dict_object_detect[key]
                state_dict_object_pose[key].requires_grad = False

        return self.model

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=True, cache_img=False
        ):
            from yolox.data import (
                LMODataset,
                YCBVDataset,
                TrainTransform,
                YoloBatchSampler,
                DataLoader,
                InfiniteSampler,
                MosaicDetection,
                worker_init_reset_seed,
            )
            from yolox.utils import (
                wait_for_the_master,
                get_local_rank,
            )

            local_rank = get_local_rank()

            with wait_for_the_master(local_rank):
                if self.data_set == "lm" or self.data_set == "lmo":
                    base_dir = "lm" if self.data_set == "lm" else "lmo"
                    dataset = LMODataset(
                            data_dir=self.data_dir,
                            json_file=self.train_ann,
                            img_size=self.input_size,
                            preproc=TrainTransform(
                                max_labels=50,
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob,
                                object_pose=self.object_pose),
                            cache=cache_img,
                            object_pose=self.object_pose,
                            base_dir=base_dir
                        )
                elif self.data_set == "ycbv":
                    dataset = YCBVDataset(
                            data_dir=self.data_dir,
                            json_file=self.train_ann,
                            img_size=self.input_size,
                            preproc=TrainTransform(
                                max_labels=50,
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob,
                                object_pose=self.object_pose),
                            cache=cache_img,
                            object_pose=self.object_pose
                        )

            dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    object_pose=self.object_pose),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )

            self.dataset = dataset

            if is_distributed:
                batch_size = batch_size // dist.get_world_size()

            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

            batch_sampler = YoloBatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=False,
                mosaic=not no_aug,
            )

            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
            dataloader_kwargs["batch_sampler"] = batch_sampler

            # Make sure each process has different random seed, especially for 'fork' method.
            # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
            dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

            train_loader = DataLoader(self.dataset, **dataloader_kwargs)

            return train_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False, visualize=False):
            from yolox.evaluators import ObjectPoseEvaluator

            val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
            output_dir = os.path.join(self.output_dir, self.exp_name)
            evaluator = ObjectPoseEvaluator(
                dataloader=val_loader,
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
                visualize=self.visualize,
                output_dir=output_dir
                )

            return evaluator