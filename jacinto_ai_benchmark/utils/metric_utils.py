import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def confusion_matrix(cmatrix, output, target, num_classes):
    output = output.flatten()
    target = target.flatten()
    mask = (target>=0) & (target<num_classes)
    merged = target[mask].astype(int) * num_classes + output[mask].astype(int)
    hist = np.bincount(merged, minlength=num_classes**2)
    hist = hist.reshape(num_classes, num_classes)
    cmatrix = cmatrix + hist if (cmatrix is not None) else hist
    return cmatrix


def segmentation_accuracy(cmatrix, multiplier=100.0):
    eps = np.finfo(np.float32).eps
    intersection = np.diag(cmatrix)
    union = np.sum(cmatrix, axis=0) + np.sum(cmatrix, axis=1) - intersection
    iou = intersection / (union + eps)
    mean_iou = np.nanmean(iou)
    metric = {'accuracy-mean-iou%':mean_iou*multiplier}
    return metric
