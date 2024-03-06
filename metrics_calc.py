import torch
import numpy as np
from sklearn.metrics import average_precision_score
import warnings

def metrics_sklearn(true_boxes, true_labels, pred_boxes, pred_labels, confidences, return_curve = False, verbose = False):
    # for precision-recall-curve
    scores_list, gt_classes_list, statistics = [], [], {'tps': 0, 'tp_ious': [], 'fps': 0, 'fns': 0}
    iou_thresholds = [0.5] # 0.5, 0.75, 0.9
    
    # iterate over each image
    for tboxes, tlabels, pboxes, plabels, confs in zip(true_boxes, true_labels, pred_boxes, pred_labels, confidences):    
        new_scores, new_gt_classes, new_statistics = get_scores_and_classes(tboxes, pboxes, confs, iou_thresholds)
        scores_list += new_scores
        gt_classes_list += new_gt_classes
        for key in statistics.keys():
            statistics[key] += new_statistics[key]

        assert len(scores_list) == len(gt_classes_list)
    
    if len(scores_list) > 0:
        precision, recall, thresholds = precision_recall_curve(gt_classes_list, scores_list, fns=statistics['fns'])
        average_precision = average_precision_score(gt_classes_list, scores_list)
    else:
        average_precision = 0.

    if verbose:
        if average_precision > 0.5:
            print(f"[verbose call] average_precision={average_precision:.3e}, TPs: {statistics['tps']}, FPs: {statistics['tps']}, FNs: {statistics['fns']}")

    if return_curve:
        from torchmetrics.utilities.plot import plot_curve
        fig, ax = plot_curve(
            (torch.tensor(recall), torch.tensor(precision)), score=average_precision, ax=None, label_names=("Recall", "Precision"), name='Precision-Recall-Curve'
        )
        return average_precision, statistics, (fig, ax)
    
    return average_precision, statistics


def precision_recall_curve(
    y_true, probas_pred, fns=0, *, pos_label=1, sample_weight=None, drop_intermediate=False
):
    """
    source implementation from sklearn
    """
    classes = np.unique(y_true)
    assert (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [0]) or np.array_equal(classes, [1])), f'Failure with given classes: {classes.tolist()}'
    
    from sklearn.metrics._ranking import _binary_clf_curve
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    if drop_intermediate and len(fps) > 2:
        # Drop thresholds corresponding to points where true positives (tps)
        # do not change from the previous or subsequent point. This will keep
        # only the first and last point for each tps value. All points
        # with the same tps value have the same recall and thus x coordinate.
        # They appear as a vertical line on the plot.
        optimal_idxs = np.where(
            np.concatenate(
                [[True], np.logical_or(np.diff(tps[:-1]), np.diff(tps[1:])), [True]]
            )
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    ps = tps + fps
    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / (tps[-1] + fns)

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def get_scores_and_classes(tboxes, pboxes, confs, iou_thresholds):
    new_scores, new_gt_classes = [], []
    tps, fps, fns = 0, 0, 0
    ious = np.zeros((len(pboxes), len(tboxes)), dtype=np.float32)
    scores = np.zeros((len(pboxes), len(tboxes)), dtype=np.float32)

    if tboxes.shape[0]==0:
        app_list = confs.tolist()
        new_scores += app_list
        new_gt_classes += [0]*len(app_list)
        assert len(new_scores) == len(new_gt_classes)
        return new_scores, new_gt_classes, {'tps': 0, 'tp_ious': [], 'fps': len(app_list), 'fns': 0}
    if pboxes.shape[0]==0:
        assert len(new_scores) == len(new_gt_classes)
        return new_scores, new_gt_classes, {'tps': 0, 'tp_ious': [], 'fps': 0, 'fns': tboxes.shape[0]}
    
    for ip, p_box in enumerate(pboxes):
        for it, gt_box in enumerate(tboxes):
            iou_ = calc_iou(p_box, gt_box)
            if iou_ > 0.:
                assert confs[ip] > 0.
                ious[ip, it] = iou_
                scores[ip, it] = confs[ip]
    
    sum_over_ious = np.sum(ious, axis=0)
    no_match_gts = np.argwhere(sum_over_ious==0)
    fns = no_match_gts.shape[0]

    max_ious = np.max(ious, axis=0)

    valid_iou_coordinates = np.argwhere((max_ious == ious) & (max_ious > iou_thresholds[0]))

    
    scores_tps = scores[valid_iou_coordinates[:, 0], valid_iou_coordinates[:, 1]]
    ious_tps = ious[valid_iou_coordinates[:, 0], valid_iou_coordinates[:, 1]].tolist()
    # assign zeros after extraction
    scores[valid_iou_coordinates[:, 0], valid_iou_coordinates[:, 1]] = 0

    new_scores += [float(val) for val in scores_tps] # true positives
    new_gt_classes += [1]*scores_tps.shape[0]
    tps = scores_tps.shape[0]

    remaining_score_indizes = np.nonzero(scores)
    remaining_scores = scores[remaining_score_indizes].tolist()

    new_scores += remaining_scores
    new_gt_classes += [0]*len(remaining_scores)
    fps = len(remaining_scores)

    return new_scores, new_gt_classes, {'tps': tps, 'tp_ious': ious_tps, 'fps': fps, 'fns': fns}


def calc_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    box1 (numpy array): [x1, y1, x2, y2] coordinates of the first bounding box.
    box2 (numpy array): [x1, y1, x2, y2] coordinates of the second bounding box.

    Returns:
    float: IoU value.
    """
    # Calculate the coordinates of the intersection rectangle
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Union area by adding the areas of both boxes and subtracting the intersection area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou_value = intersection_area / union_area

    return iou_value

