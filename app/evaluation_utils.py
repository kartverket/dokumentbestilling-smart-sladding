import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
import model_main
import ast
from pdf2image import convert_from_path, convert_from_bytes
import fitz
import model_utils
from url_utils import api_base_url

def match_bboxes(true_bboxes, predicted_bboxes, iou_threshold=0.2):
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

    for true_bbox in true_bboxes:
        # If negative height or width, convert to positive and adjust left and top
        if true_bbox[0] < 0:
            true_bbox[0] = abs(true_bbox[0])
            true_bbox[3] = true_bbox[3] - true_bbox[0]
        if true_bbox[1] < 0:
            true_bbox[1] = abs(true_bbox[1])
            true_bbox[2] = true_bbox[2] - true_bbox[1]

    num_true_bboxes = len(true_bboxes)
    num_pred_bboxes = len(predicted_bboxes)

    # If there are no true bboxes or predicted bboxes, return matched list with None values and all predicted bboxes as unmatched

    # If there are no true bboxes or predicted bboxes, return matched list with None values and all predicted bboxes as unmatched
    if num_true_bboxes == 0 or num_pred_bboxes == 0:
        matched_boxes = [[true_bbox, None, 0] for true_bbox in true_bboxes]
        unmatched_preds = predicted_bboxes
        return matched_boxes, unmatched_preds, {'TP': 0, 'FP': num_pred_bboxes, 'FN': num_true_bboxes}
    
    
    # Calculate IOU matrix
    iou_matrix = model_utils.calculate_iou(true_bboxes, predicted_bboxes)
    
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

def convert_pdf_bytes_to_images(pdf_bytes):
    """
    Convert a PDF file to a list of images.

    Parameters:
    pdf_bytes (bytes): The PDF file content in bytes.

    Returns:
    list: A list of images.
    tuple: A tuple containing the width and height of the images.
    """
    images = convert_from_bytes(pdf_bytes)
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


def get_true_boxes_from_docid(labels_df, docid, folder_path):
    """
    Get the images and bounding boxes for a document from the document id(dokument-ident).

    Parameters:
    labels_df (pd.DataFrame): A DataFrame containing the labels.
    docid (str): The document ID.
    folder_path (str): The path to the folder containing the documents.

    Returns:
    list: A list of images.
    list: A list of scaled bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    """
    # Get the bounding boxes for the document
    row = labels_df.loc[labels_df['dokument_nr_embete'] == docid, 'bbs']

    # Get the bounding boxes from the row and interpret as a list
    bbs = ast.literal_eval(row.iloc[0])

    #Get the page_count for the document in valideringssett
    filename = folder_path + f"{docid}.pdf"
    page_count = get_pdf_pagecount(filename)

    # Add empty lists for pages without bounding boxes
    for i in range(len(bbs), page_count):
        bbs.append([])
    
    # Get the dimensions of the downloaded pdf file and the local pdf file
    dimensions = get_pdf_dimensions(filename)
    images, dimensions_hq = convert_pdf_path_to_images(filename)

    # Calculate the ratio between the high quality images and the images from pdf2image
    dimension_ratio = dimensions_hq[0]/dimensions[0]

    # Scale the bounding boxes
    bbs_scaled = [[scale_bounding_box(bb, dimension_ratio) for bb in page] for page in bbs] 
    return images, bbs_scaled


def visualize_bounding_boxes(images, predicted_bboxes, true_bboxes):
    """
    Draw bounding boxes on images.

    Parameters:
    images (list): A list of images.
    predicted_bboxes (list): A list of predicted bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    true_bboxes (list): A list of labelled bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].

    Returns:
    list: A list of images with bounding boxes.
    """
    images_with_bb = []
    for i, image in enumerate(images):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)

        # Combine true and predicted bounding boxes with labels
        combined_bbs = []
        if len(true_bboxes) > 0:
            combined_bbs += [(bb, 'True') for bb in true_bboxes[i]]
        if len(predicted_bboxes) > 0:
            combined_bbs += [(bb, 'Predicted') for bb in predicted_bboxes[i]]

        # Plot each bounding box with appropriate label and color
        for bb, label in combined_bbs:
            if label == 'True':
                edgecolor = 'green'
                linestyle = '-'
                legend_label = 'True'
            if label == 'Predicted':
                edgecolor = 'red'
                linestyle = '--'
                legend_label = 'Predicted'

            rect = plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], linewidth=2, linestyle=linestyle,
                                 edgecolor=edgecolor, facecolor="none", label=legend_label)
            ax.add_patch(rect)

        # Creating a legend with unique handles
        handles, labels = ax.get_legend_handles_labels()
        unique = {(h.get_edgecolor(), l): h for h, l in zip(handles, labels)}.values()
        if unique:
            ax.legend(unique, [h.get_label() for h in unique], loc='upper right')

        images_with_bb.append(fig)

    return images_with_bb

def visualize_bounding_boxes_detailed(images, all_bbs, true_bbs, pred_key, pred_regex, pred_hjemmel, show=False):
    """
    Visualize bounding boxes on images.

    Parameters:
    images (list): A list of images.
    all_bbs (list): A list of all bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    true_bbs (list): A list of labelled bounding boxes separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    pred_key (list): A list of predicted bounding boxes separated by keyword search by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].
    pred_regex (list): A list of predicted bounding boxes by regex pattern separated by page [[bb1, bb2, ...], [bb1, bb2, ...], ...] where bb is [height, width, x, y].

    Returns:
    list: A list of images with bounding boxes.
    """
    images_with_bb = []
    for i, image in enumerate(images):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)

        # Combine true and predicted bounding boxes with labels
        combined_bbs = []
        combined_bbs += [(bb, 'All') for bb in all_bbs[i]]
        if i < len(true_bbs):
            combined_bbs += [(bb, 'True') for bb in true_bbs[i]]
        # if i < len(pred_bbs):
        #     combined_bbs += [(bb, 'Predicted') for bb in pred_bbs[i]]
        if i < len(pred_key):
            combined_bbs += [(bb, 'Predicted keyword') for bb in pred_key[i]]
        if i < len(pred_regex):
            combined_bbs += [(bb, 'Predicted regex') for bb in pred_regex[i]]
        if i < len(pred_hjemmel):
            combined_bbs += [(bb, 'Predicted hjemmel') for bb in pred_hjemmel[i]]

        # Plot each bounding box with appropriate label and color
        for bb, label in combined_bbs:
            if label == 'All':
                edgecolor = 'gray'
                linestyle = '-'
                legend_label = 'All (label)'
            if label == 'True':
                edgecolor = 'green'
                linestyle = '-'
                legend_label = 'True (label)'
            if label == 'Predicted regex':
                edgecolor = 'red'
                linestyle = ':'
                legend_label = 'Predicted regex (label)'
            if label == 'Predicted keyword':
                edgecolor = 'blue'
                linestyle = ':'
                legend_label = 'Predicted keyword (label)'
            if label == 'Predicted hjemmel':
                edgecolor = 'purple'
                linestyle = ':'
                legend_label = 'Predicted hjemmel (label)'

            rect = plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], linewidth=2, linestyle=linestyle,
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


def get_metrics_and_cm(total_tp, total_fp, total_fn):
    """
    Calculate the precision, recall, and F1 score from the total number of true positives, false positives, and false negatives. Plot the metrics in a confusion matrix.
    
    Parameters:
    total_tp (int): The total number of true positives.
    total_fp (int): The total number of false positives.
    total_fn (int): The total number of false negatives.
    """

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('Accuracy:', total_tp/(total_tp + total_fp + total_fn))
    print('Percent of true positives:', total_tp/(total_tp + total_fn))

    # Define the confusion matrix
    conf_matrix = np.array([[total_tp, total_fp], 
                            [total_fn, 0]])

    # Labels for each cell
    group_names = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
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


def get_predicted_boxes_on_doc(docid, cache_path):
    """
    Get the predicted bounding boxes for a document.
    Need to be on kartverket VPN to download documents and therefore also to run this function.

    Parameters:
    docid (str): The document ID. <dokumener_aar>_<dokument_nr>_<embete>
    base_url (str): The base URL to the document.

    Returns:
    list: A list of predicted bounding boxes.
    """

    # Download the PDF file

    # Load document bytes from dokument.pdf
    try:
        with open(f'valideringssett/all_documents/{docid}.pdf', 'rb') as f:
            pdf_bytes = f.read()
    except FileNotFoundError:
        pdf_bytes = model_utils.download_pdf(f'{api_base_url()}intern/pantebok/gjenpart/{docid}?attestering=false')

    # Get the predicted bounding boxes
    (images,
     all_text,
     model_bbs,
     predicted_boxes,
     predicted_keyword,
     predicted_regex,
     predicted_hjemmel,
     image_dimensions) = model_main.extended_model(
        pdf_bytes,
        save_bbs_path=f'{cache_path}/bbs/{docid}',
        save_text_path=f'{cache_path}/texts/{docid}',
        debug_print=True,
        only_first_page=True
    )

    # Get the dimensions of the PDF file
    predicted_boxes_scaled = model_utils.scale_and_pad_all_bounding_boxes(predicted_boxes, ratio=1)

    # Get the images with the bounding boxes painted on them
    images_with_bb = visualize_bounding_boxes_detailed(images, model_bbs, [], predicted_keyword, predicted_regex, predicted_hjemmel, show=True)

    # Save the images with the bounding boxes
    for i, image in enumerate(images_with_bb):
        image.savefig(f'app/predicted_boxes/{docid}_page_{i}.png')

    return predicted_boxes_scaled


if __name__ == "__main__":
    cache_path = 'valideringssett/results/1'

    # Run get_predicted_boxes_on_doc to get the predicted bounding boxes for a document. This is a great debugging tool.

    ### Examples to test hjemmel_search ###
    #get_predicted_boxes_on_doc('2022_446714_200', cache_path)
    #get_predicted_boxes_on_doc('1993_4626_86', cache_path)
    #get_predicted_boxes_on_doc('1992_2274_65', cache_path)
    #get_predicted_boxes_on_doc('1991_59180_105', cache_path)
    #get_predicted_boxes_on_doc('2012_343635_200', cache_path)
