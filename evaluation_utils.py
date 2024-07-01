
import numpy as np
import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def match_bboxes(true_bboxes, predicted_bboxes, iou_threshold=0.5):
    """
    Match predicted bounding boxes to true bounding boxes based on Intersection over Union (IOU) value.

    Args:
    true_bboxes: np.array, shape=(n, 4), n is the number of true bboxes in one page, 4 is the number of bbox coordinates. 
    predicted_bboxes: np.array, shape=(m, 4), m is the number of predicted bboxes in one page, 4 is the number of bbox coordinates.
    iou_threshold: float, the minimum IOU value for a predicted bbox to be considered a match for a true bbox.

    Returns:
    matched_boxes: list of lists, each list contains an array of true bbox coordinates, an array of predicted bbox coordinates and their IOU value.
    unmatched_preds: np.array, shape=(m, 4), m is the number of unmatched predicted bboxes in one page, 4 is the number of bbox coordinates.
    metrics: dict, contains the number of True Positives (TP), False Positives (FP) and Predicted Score (PS).
    """

    num_true_bboxes = true_bboxes.shape[0]
    num_pred_bboxes = predicted_bboxes.shape[0]

    # If there are no true bboxes or predicted bboxes, return matched list with None values and all predicted bboxes as unmatched

    if num_true_bboxes == 0 or num_pred_bboxes == 0:
        matched_boxes = [[true_bbox, None, 0] for true_bbox in true_bboxes]
        unmatched_preds = predicted_bboxes.copy()
        
        return matched_boxes, unmatched_preds
    
    # Calculate IOU matrix

    iou_matrix = box_iou(torch.tensor(true_bboxes), torch.tensor(predicted_bboxes)).numpy()
    
    # Find the best matches

    matched_boxes = []
    unmatched_true = true_bboxes.copy()
    unmatched_preds = predicted_bboxes.copy()

    while np.any(iou_matrix > iou_threshold):
        best_match = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        true_index, pred_index = best_match

        matched_boxes.append([true_bboxes[true_index], predicted_bboxes[pred_index], iou_matrix[true_index, pred_index]])
        unmatched_preds = np.delete(unmatched_preds, pred_index, axis=0)
        unmatched_true = np.delete(unmatched_true, true_index, axis=0)

        iou_matrix[true_index, :] = 0
        iou_matrix[:, pred_index] = 0
    
    # Calculate the number of True Positives, False Positives, precision, recall
    tp = len(matched_boxes)
    fp = len(unmatched_preds)
    fn = len(unmatched_true)

    # Add the unmatched true bboxes to the matched list

    for unmatch_t in unmatched_true:
        matched_boxes.append([unmatch_t, None, 0])
    
    metrics = {'TP': tp, 'FP': fp, 'FN': fn} # True Positives, False Positives, precision, recall

    return matched_boxes, unmatched_preds, metrics

def metrics_perdocument(metrics_list):
    """
    Calculate the aggregated metrics for each document in a dataset.
    
    Args:
    metrics_list (List[Dict[str, int]]): A list of dictionaries, where each dictionary contains
                                         'TP', 'FP', and 'FN' for a document.
    
    Returns:
    Dict[str, float]: A dictionary containing aggregated 'TP', 'FP', 'FN', 'precision', and 'recall'.
    """
    
    # Aggregate true positives, false positives, and false negatives
    TP = sum(metrics.get('TP', 0) for metrics in metrics_list)
    FP = sum(metrics.get('FP', 0) for metrics in metrics_list)
    FN = sum(metrics.get('FN', 0) for metrics in metrics_list)
    
    # Calculate precision and recall, handle division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': precision, 'recall': recall, 'F1': F1}