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



def get_metrics(pred_boxes, gt_boxes, class_name, dist_th):
    all_gt_boxes = []
    for sample_token, boxes in gt_boxes.items():
        all_gt_boxes.extend(boxes)
    npos = len([1 for box in all_gt_boxes if box['detection_name'] == class_name])
    if npos == 0:
        return {}
    all_pred_boxes = []
    for sample_token, boxes in pred_boxes.items():
        all_pred_boxes.extend(boxes)
    
    pred_boxes_class = [box for box in all_pred_boxes if box['detection_name'] == class_name]
    pred_confs = [box['detection_score'] for box in pred_boxes_class]
    
    sort_idx = sorted(range(len(pred_confs)), reverse=True, key=lambda k: pred_confs[k])
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}
    taken = set()  # Initially no gt bounding box is matched.
    for idx in sort_idx:
        pred_box = pred_boxes_class[idx]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box['sample_token']]):

            # Find closest match among ground truth boxes
            if gt_box['detection_name'] == class_name and not (pred_box['sample_token'], gt_idx) in taken:
                this_distance = center_distance(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box['sample_token'], match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box['detection_score'])

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box['sample_token']][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box['detection_score'])

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box['detection_score'])

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return {}

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]
    max_recall_ind = -1
    for conf in match_data['conf']:
        if conf == 0:
            break
        max_recall_ind +=1
    
    return dict(recall=rec,
                precision=prec,
                confidence=conf,
                max_recall_ind = max_recall_ind,
                trans_err=match_data['trans_err'],
                vel_err=match_data['vel_err'],
                scale_err=match_data['scale_err'],
                orient_err=match_data['orient_err'],
                attr_err=match_data['attr_err'])
    
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
