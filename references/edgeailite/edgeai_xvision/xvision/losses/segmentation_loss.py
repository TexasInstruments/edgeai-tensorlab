#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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

from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import torch
from .loss_utils import *
from edgeai_torchmodelopt import xnn

__all__ = ['segmentation_loss', 'segmentation_metrics', 'SegmentationMetricsCalc']


def cross_entropy2d(input, target, weight=None, ignore_index=None, size_average=True):
    #nll_loss expects long tensor target
    target = target.long()

    # 1. input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # 2. log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)

    # 3. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    #assert torch.all((target>=0) & ((target<19) | (target == ignore_index))), 'target range problem'

    # 4. target: (n*h*w,)
    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, ignore_index=ignore_index, weight=weight, size_average=False)
    if size_average:
        # loss /= mask.sum().data[0]
        sum_val = mask.data.sum().float() if mask.numel()>0 else 0
        loss = (loss/sum_val) if (sum_val!=0) else (loss*np.float32(0.0))
    return loss


class SegmentationLoss(torch.nn.Module):
    def __init__(self, *args, ignore_index = 255, weight=None, enable_fp16=False, **kwargs):
        super().__init__()
        self.enable_fp16 = enable_fp16
        if weight is None:
            self.weight = None
        else:
            self.register_buffer('weight', torch.FloatTensor(weight))
        #
        self.ignore_index = ignore_index
        self.is_avg = False
    #
    @xnn.utils.auto_fp16
    def forward(self, input_img, input, target):
        weight = self.weight if (self.weight is not None and np.random.random() < 0.20) else None
        loss = cross_entropy2d(input, target, weight, ignore_index=self.ignore_index)
        return loss
    def info(self):
        return {'value':'loss', 'name':'CrossEntropyLoss', 'is_avg':self.is_avg}
    def clear(self):
        return
    @classmethod
    def args(cls):
        return ['weight','enable_fp16']
#
segmentation_loss = SegmentationLoss


class SegmentationMetricsCalc():
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def __call__(self, label_preds, label_trues):
        return self._update(label_preds, label_trues)

    @staticmethod
    def _fast_hist(label_pred, label_true, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def _update(self, label_preds, label_trues):
        if type(label_trues) == torch.Tensor:
            label_trues = label_trues.cpu().long().numpy()
            if label_preds.shape[1] > 1:
                label_preds = label_preds.max(1)[1].cpu().numpy()
            else:
                label_preds = label_preds.cpu().numpy()

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lp.flatten(), lt.flatten(), self.n_classes)

        return self._get_scores()

    def _get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)

        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        acc_cls = tp / (sum_a1 + np.finfo(np.float32).eps)
        acc_cls = np.nanmean(acc_cls)

        iou = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iou = np.nanmean(iou)

        freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        iou = iou*100
        mean_iou = mean_iou*100
        acc = acc*100
        acc_cls = acc_cls*100
        fwavacc = fwavacc*100

        cls_iou = dict(zip(range(self.n_classes), iou))
        return {'MeanIoU': mean_iou, 'OverallAcc': acc,
                'MeanAcc': acc_cls, 'FreqWtAcc': fwavacc,
                'ClsIoU':cls_iou}

    def clear(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class SegmentationMetrics(torch.nn.Module):
    def __init__(self, *args, num_classes=None, enable_fp16=False, **kwargs):
        super().__init__()
        self.enable_fp16 = enable_fp16
        self.metrics_calc = SegmentationMetricsCalc(num_classes)
        # the output is an using the confusion matrix accumulated so far, after clear() was called.
        self.is_avg = True
    #
    @xnn.utils.auto_fp16
    def forward(self, input_img, input, target):
        is_cuda = input.is_cuda or target.is_cuda
        metrics = self.metrics_calc(input, target)
        metrics = metrics['MeanIoU']
        metrics = torch.FloatTensor([metrics])
        metrics = metrics.cuda() if is_cuda else metrics
        return metrics
    def forward_all(self, input, target):
        metrics = self.metrics_calc(input, target)
        return metrics
    def clear(self):
        self.metrics_calc.clear()
        return
    def info(self):
        return {'value':'accuracy', 'name':'MeanIoU', 'is_avg':self.is_avg, 'confusion_matrix':self.metrics_calc.confusion_matrix}
    @classmethod
    def args(cls):
        return ['num_classes','enable_fp16']
#
segmentation_metrics = SegmentationMetrics
