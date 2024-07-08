import numpy as np
import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import PyPDF2
import pandas as pd
import model_utils
import model_main
import ast
import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import fitz

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

    # If there are no true bboxes or predicted bboxes, return matched list with None values and all predicted bboxes as unmatched
    if num_true_bboxes == 0 or num_pred_bboxes == 0:
        matched_boxes = [[true_bbox, None, 0] for true_bbox in true_bboxes]
        unmatched_preds = predicted_bboxes
        return matched_boxes, unmatched_preds, {'TP': 0, 'FP': num_pred_bboxes, 'FN': num_true_bboxes}
    
    
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

def get_pdf_dimensions_from_byte_file(pdf_bytes):
    """
    Get the dimensions of each page in a PDF file.

    Parameters:
    pdf_bytes (bytes): The PDF file content in bytes.

    Returns:
    list: A list of tuples where each tuple contains the width and height of a page.
    """
    # Open the PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    page = pdf_document.load_page(0)
    rect = page.rect
    return (rect.width, rect.height)

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


def convert_pdf_path_to_images(pdf_path):
    """
    Convert a PDF file to a list of images.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of images.
    tuple: A tuple containing the width and height of the images.
    """
    images = convert_from_path(pdf_path)
    width, height = images[0].size
    dimensions = (width, height)
    return images, dimensions


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

def scale_all_bounding_boxes(bounding_boxes, ratio):
    """
    Scales all bounding boxes in a list by a given ratio.

    Parameters:
    bounding_boxes (list): A list of bounding boxes.
    ratio (float): The scaling ratio.

    Returns:
    list: A new list of bounding boxes scaled by the given ratio.
    """

    scaled_boxes = []

    for page in bounding_boxes:
        page_boxes = []
        for bbox in page:
            if bbox:
                page_boxes.append(scale_bounding_box(bbox, ratio))
        scaled_boxes.append(page_boxes)

    return scaled_boxes


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
    # Get the bounding boxes for the document
    row = labels_df.loc[labels_df['dokument_nr_embete'] == docid, 'bounding_boxes']

    # Get the bounding boxes from the row and interpret as a list
    bbs = ast.literal_eval(row.iloc[0])

    #Get the page_count for the document in valideringssett
    filename = f"{docid}.pdf"
    page_count = get_pdf_pagecount(filename)

    # Add empty lists for pages without bounding boxes
    for i in range(len(bbs), page_count):
        bbs.append([])
    
    # Get the dimensions of the downloaded pdf file and the local pdf file
    dimensions = get_pdf_dimensions(filename)
    images, dimensions_hq = model_utils.convert_pdf_path_to_images(filename)

    # Calculate the ratio between the high quality images and the images from pdf2image
    dimention_ratio = dimensions_hq[0]/dimensions[0]

    # Scale the bounding boxes
    bbs = [[scale_bounding_box(bb, dimention_ratio) for bb in page] for page in bbs]
    return images, bbs


def visualize_bounding_boxes(images, true_bbs, pred_bbs, show=False):
    """
    Visualize bounding boxes on images.

    Parameters:
    images (list): A list of images.
    true_bbs (list): A list of labelled bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    pred_bbs (list): A list of predicted bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    """
    images_with_bb = []
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

        images_with_bb.append(fig)
    
    if show:
        plt.show()
    #empty the plot 
    plt.close('all')

    return images_with_bb

def test_and_visualize_doc(doc_id, visualize=False):
    pdf_path = f'../valideringssett/dokumenter/{doc_id}.pdf'

    organized_labels_path = pd.read_csv("../valideringssett/organized_data.csv")

    images_true, true_boxes = get_images_and_bb_from_docid(organized_labels_path, doc_id)

    images_pred, predicted_boxes = model_main.main(pdf_path)
    
    images_with_bbs = visualize_bounding_boxes(images_true, true_boxes, predicted_boxes, show=visualize)
    
    return images_with_bbs


def get_metrics_and_cm(total_tp, total_fp, total_fn):

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('Accuracy:', total_tp/(total_tp + total_fp + total_fn))

    # Define the confusion matrix
    conf_matrix = np.array([[total_tp, total_fp], 
                            [total_fn, 0]])

    # Labels for each cell
    group_names = ['True Positive', 'False Positive', 'False Negative','True Negative']
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten() / np.sum(conf_matrix)]
    labels = (np.asarray(["{}\n{}\n{}".format(name, count, pct) for name, count, pct in zip(group_names, group_counts, group_percentages)])).reshape(2,2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')

    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        xticklabels=['Positives','Negatives'], 
        yticklabels=['Positives','Negatives'],
        title='Confusion Matrix')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, labels[i, j],
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    plt.ylabel('Predicted labels')
    plt.xlabel('Actual labels')
    plt.tight_layout()
    plt.show()


##Avhengig av config..

## RESULTS ALL DOCUMENTS
#Total TP: 686, Total FP: 9, Total FN: 823

## RESULTS ELEKTRONISK TINGLYST
#Total TP: 162, Total FP: 0, Total FN: 0