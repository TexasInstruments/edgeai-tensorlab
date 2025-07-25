import numpy as np


def center_distance(gt_box, pred_box):
    return np.linalg.norm(np.array(gt_box['translation']) - np.array(pred_box['translation']))


def velocity_l2(gt_box, pred_box):
    return np.linalg.norm(np.array(gt_box['velocity']) - np.array(pred_box['velocity']))

def yaw_diff(gt_box, pred_box, period=2 * np.pi):
    diff = (gt_box['yaw'] - pred_box['yaw'])
    diff = (diff + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  
    return np.abs(diff)


def attr_acc(gt_box, pred_box):
    if gt_box['attribute_name'] == 'None':
        return np.nan
    else:
        return float(gt_box['attribute_name'] == pred_box['attribute_name'])


def scale_iou(gt_box, pred_box) -> float:
    # Validate inputs.
    sa_size = np.array(gt_box['size'])
    sr_size = np.array(pred_box['size'])
    assert all(sa_size > 0), 'Error: gt_box sizes must be >0.'
    assert all(sr_size > 0), 'Error: pred_box sizes must be >0.'
    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    gt_volume = np.prod(sa_size)
    pred_volume = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = gt_volume + pred_volume - intersection  # type: float
    iou = intersection / union
    return iou

def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def calc_ap(md, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """
    if len(md) == 0:
        return 0.0
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.get('precision'))
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    result = float(np.mean(prec)) / (1.0 - min_precision)
    return max(0.0, min(1.0, result))  # Clip to 0-1 range.


def calc_tp(md, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """
    if len(md) == 0:
        return 1.0
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.get('max_recall_ind',-1)  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind :
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        result = float(np.mean(md.get(metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
        return max(0.0, min(1.0, result))
