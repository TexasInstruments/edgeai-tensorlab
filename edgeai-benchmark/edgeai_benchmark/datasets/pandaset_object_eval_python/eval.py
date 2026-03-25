import numpy as np

def pandaset_evaluate_metrics(pred_boxes, gt_boxes, classes, dist_thrs, dist_thr_tp):
    from . import utils 
    MEAN_AP_WEIGHT = 5
    MIN_PRECISION, MIN_RECALL = 0.1, 0.1
    metric_data_lists = {}
    TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
    mean_dist_aps = {}
    for name in classes:
        metric_data_list = metric_data_lists[name] = {}
        for dist_th in dist_thrs:
            metric_data_list[dist_th] = metric_data = get_metrics(pred_boxes, gt_boxes, name, dist_th)
            # if len(metric_data) == 0:
            #     continue
            ap = utils.calc_ap(metric_data, MIN_RECALL, MIN_PRECISION)
            metric_data['ap'] = ap
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[dist_thr_tp]
            # if len(metric_data) == 0:
            #     continue
            metric_data[metric_name] = utils.calc_tp(metric_data, MIN_RECALL, metric_name)

        mean_dist_aps[name] ={'ap': np.mean(np.array([metric_data['ap'] for metric_data in metric_data_list.values() if metric_data]))}
    
    mean_ap = np.mean([ap_dict['ap'] for ap_dict in mean_dist_aps.values()])
    tp_errors = {}
    label_tp_errors = {}
    for metric_name in TP_METRICS:
        class_errors = []
        for detection_name in classes:
            metric_data = metric_data_lists[detection_name][dist_thr_tp]
            # if len(metric_data) ==0:
            #     continue
            class_errors.append(metric_data.get(metric_name, float('nan')))
            if detection_name not in label_tp_errors:
                label_tp_errors[detection_name] = {metric_name:metric_data.get(metric_name, float('nan'))}
            else:
                label_tp_errors[detection_name][metric_name] = metric_data.get(metric_name, float('nan'))

        tp_errors[metric_name] = float(np.nanmean(class_errors))
    
    tp_scores = {}
    for metric_name in TP_METRICS:
        # We convert the true positive errors to "scores" by 1-error.
        score = 1.0 - tp_errors.get(metric_name, float('nan'))

        # Some of the true positive errors are unbounded, so we bound the scores to min 0.
        score = max(0.0, score)

        tp_scores[metric_name] = score

    total = float(MEAN_AP_WEIGHT * mean_ap + np.sum(list(tp_scores.values())))

    # Normalize.
    nd_score = total / float(MEAN_AP_WEIGHT + len(tp_scores.keys()))
    return dict(mean_ap=mean_ap, mean_dist_aps=mean_dist_aps, label_tp_errors=label_tp_errors, tp_errors=tp_errors, tp_scores=tp_scores, nd_score=nd_score, metric_data_lists=metric_data_lists)


def get_metrics(pred_boxes, gt_boxes, class_name, dist_th):
    from . import utils 
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
                this_distance = utils.center_distance(gt_box, pred_box)
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

            match_data['trans_err'].append(utils.center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(utils.velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - utils.scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(utils.yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - utils.attr_acc(gt_box_match, pred_box))
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
            tmp = utils.cummean(np.array(match_data[key]))

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