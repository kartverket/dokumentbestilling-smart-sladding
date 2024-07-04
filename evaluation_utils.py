
import numpy as np
import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PyPDF2
import pandas as pd
import requests
import model_utils
import model_main
import ast
<<<<<<< HEAD
import os
=======
>>>>>>> main

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

    num_true_bboxes = len(true_bboxes)
    num_pred_bboxes = len(predicted_bboxes)

    # If there are no true bboxes or predicted bboxes, return matched list with None values and all predicted bboxes as unmatched

<<<<<<< HEAD
    # If there are no true bboxes or predicted bboxes, return matched list with None values and all predicted bboxes as unmatched
    if num_true_bboxes == 0 or num_pred_bboxes == 0:
        matched_boxes = [[true_bbox, None, 0] for true_bbox in true_bboxes]
        unmatched_preds = predicted_bboxes
        return matched_boxes, unmatched_preds, {'TP': 0, 'FP': num_pred_bboxes, 'FN': num_true_bboxes}
=======
    if num_true_bboxes == 0:
        matched_boxes = [[true_bbox, None, 0] for true_bbox in true_bboxes]
        unmatched_preds = predicted_bboxes.copy()
        
        return matched_boxes, unmatched_preds, {}
    
    if num_pred_bboxes == 0:
        matched_boxes = [[true_bbox, None, 0] for true_bbox in true_bboxes]
        metrics = {'TP': 0, 'FP': 0, 'FN': len(true_bboxes)}
        
        return matched_boxes, predicted_bboxes, metrics
>>>>>>> main
    
    
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

        unmatched_preds.remove(predicted_bboxes[pred_index])
        unmatched_true.remove(true_bboxes[true_index])

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


def download_and_save_pdf(url, filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)   
#url = f""
#filename = f"dokumenter/{doc}.pdf"
#download_and_save_pdf(url, filename)


def scale_bounding_box(bbox, ratio):
    """
    Scales a bounding box by a given ratio.

    Parameters:
    bbox (list): A list representing the bounding box [height, width, x, y].
    ratio (float): The scaling ratio.

    Returns:
    list: A new bounding box scaled by the given ratio.
    """
    height, width, x, y = bbox
    return [height*ratio, width*ratio, x*ratio, y*ratio]



def get_pdf_dimensions(pdf_path):
    """
    Get the dimensions of a PDF file.
    --MARK! Assuming that all pages have the same dimensions.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A tuples containing the width and height of each page in the PDF file.
    """
    pdf = PyPDF2.PdfReader(open(pdf_path, "rb"))
    media_box = pdf.pages[0].mediabox
    return (float(media_box.width),float(media_box.height))

def get_pdf_pagecount(pdf_path):
    """
    Get the number of pages in a PDF file.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    int: The number of pages in the PDF file.
    """
    pdf = PyPDF2.PdfReader(open(pdf_path, "rb"))
    return len(pdf.pages)

def get_images_and_bb_from_docid(labels_df, docid):
    """
    Get the images and bounding boxes for a document from the document id(dokument-ident).

    Parameters:
    labels_df (pd.DataFrame): A DataFrame containing the labels.
    docid (str): The document ID.

    Returns:
    list: A list of images.
    list: A list of scaled bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    """
    row = labels_df.loc[labels_df['dokument_nr_embete'] == docid, 'bounding_boxes']
    doc = docid
    bbs_string = row.iloc[0]
    bbs = ast.literal_eval(bbs_string)
<<<<<<< HEAD
    filename = f"../valideringssett/dokumenter/{doc}.pdf"
=======
    filename = f"valideringssett/dokumenter/{doc}.pdf"
>>>>>>> main
    page_count = get_pdf_pagecount(filename)
    for i in range(len(bbs), page_count):
        bbs.append([])
    dimensions = get_pdf_dimensions(filename)
    images, dimensions_hq = model_utils.convert_pdf_to_images(filename)
    # Calculate the ratio between the high quality images and the images from pdf2image
    dimention_ratio = dimensions_hq[0]/dimensions[0]
    bbs = [[scale_bounding_box(bb, dimention_ratio) for bb in page] for page in bbs]
    return images, bbs



<<<<<<< HEAD
def visualize_bounding_boxes(images, true_bbs, pred_bbs, show=False):
=======
def visualize_bounding_boxes(images, true_bbs, pred_bbs):
>>>>>>> main
    """
    Visualize bounding boxes on images.

    Parameters:
    images (list): A list of images.
    true_bbs (list): A list of labelled bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    pred_bbs (list): A list of predicted bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    """
<<<<<<< HEAD
    images_with_bb = []
=======
>>>>>>> main
    for i, image in enumerate(images):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)

        # Combine true and predicted bounding boxes with labels
        combined_bbs = []
        if i < len(true_bbs):
            combined_bbs += [(bb, 'True') for bb in true_bbs[i]]
        if i < len(pred_bbs):
            combined_bbs += [(bb, 'Predicted') for bb in pred_bbs[i]]

        # Plot each bounding box with appropriate label and color
        for bb, label in combined_bbs:
            if label == 'True':
                edgecolor = 'green'
                legend_label = 'True (label)'
            else:
                edgecolor = 'red'
                legend_label = 'Predicted (label)'

            rect = plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], linewidth=2,
                                 edgecolor=edgecolor, facecolor="none", label=legend_label)
            ax.add_patch(rect)

        # Creating a legend with unique handles
        handles, labels = ax.get_legend_handles_labels()
        unique = {(h.get_edgecolor(), l): h for h, l in zip(handles, labels)}.values()
        if unique:
            ax.legend(unique, [h.get_label() for h in unique], loc='upper right')

<<<<<<< HEAD
        images_with_bb.append(fig)
    
    if show:
        plt.show()
    #empty the plot 
    plt.close('all')

    return images_with_bb

def test_and_visualize_doc(doc_id, visualize=False):
    pdf_path = f'../valideringssett/dokumenter/{doc_id}.pdf'

    organized_labels_path = pd.read_csv("../valideringssett/organized_data.csv")
=======
        #plt.show()
        #Save the image
        plt.savefig(f'test_viz{i}')

#doc_id = "2023_73325_200"
#doc_id = "2023_73413_200"
#doc_id = "2010_923067_200"

def test_and_visualize_doc(doc_id):
    pdf_path = f'valideringssett/dokumenter/{doc_id}.pdf'

    organized_labels_path = pd.read_csv("valideringssett/organized_data.csv")
>>>>>>> main

    images_true, true_boxes = get_images_and_bb_from_docid(organized_labels_path, doc_id)

    images_pred, predicted_boxes = model_main.main(pdf_path)
<<<<<<< HEAD
    
    images_with_bbs = visualize_bounding_boxes(images_true, true_boxes, predicted_boxes, show=visualize)
    
    return images_with_bbs
=======

    visualize_bounding_boxes(images_true, true_boxes, predicted_boxes)
>>>>>>> main
