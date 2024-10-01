import os

import evaluation_utils
import model_main
import pandas as pd
import model_utils
import time


header_printing_interval = 20
header_text = f'{"Index":<5} {"Time(s)":<8} {"Document":<20} {"Pages":<6} {"TP":<3} {"FP":<3} {"FN":<3} {"Total Time(s)":<13} {"Total TP":<9} {"Total FP":<9} {"Total FN":<9} {"Precision":<10} {"Recall":<10} {"F1":<10} {"Accuracy":<10}'

def get_precision(tp, fp):
    return round(tp / (tp + fp), 2)


def get_recall(tp, fn):
    return round(tp / (tp + fn), 2)


def get_f1(precision, recall):
    return round(2 * (precision * recall) / (precision + recall), 2)


def get_accuracy(tp, fp, fn):
    return round(tp / (tp + fp + fn), 2)

def evaluate_model(current_model_version_number, document_folder, labels_csv, docids_csv, savefolder_name):
    """
    Evaluate the model on a set of documents and save the images with bounding boxes(ground truth and predicted) and the text extracted from the documents.

    Parameters:
    current_model_version_number (str): The version number of the model. Used to save the results.
    document_folder (str): The path to the folder containing the documents.
    labels_csv (str): The path to the CSV file containing the labels. Has to contain only the accepted labels(The correct labels).
    docids_csv (str): The path to the CSV file containing the document IDs.
    savefolder_name (str): The path to the folder to save the results to.
    """
    major_version = current_model_version_number.split(".")[0]
    major_versioned_savefolder_name = f'{savefolder_name}/{major_version}'
    versioned_savefolder_name = f'{major_versioned_savefolder_name}/{current_model_version_number}'
    results_path = f'{versioned_savefolder_name}/results.csv'

    total_time = time.time()

    # Load labels
    labels_df = pd.read_csv(labels_csv)
    labels_df['docid'] = labels_df['dokument_aar'].astype(str) + '_' + labels_df['dokument_nr'].astype(str) + '_' + labels_df['embete'].astype(str)

    # load document-ids
    docids_df = pd.read_csv(docids_csv)
    docids_df['docid'] = docids_df['dokument_aar'].astype(str) + '_' + docids_df['dokument_nr'].astype(str) + '_' + docids_df['embete'].astype(str)

    current_results = pd.DataFrame([], columns=['Document', 'Page', 'Time(s)', 'TP', 'FP', 'FN'])
    try:
        current_results = pd.read_csv(results_path)
        for index, row in current_results.iterrows():
            if index % header_printing_interval == 0:
                print(header_text)

            all_curr_results = current_results.head(index + 1)
            table_text = f'{index:<5} {row["Time(s)"]:<8} {row["Document"]:<20} {row["Page"]:<6} {row["TP"]:<3} {row["FP"]:<3} {row["FN"]:<3} {round(all_curr_results["Time(s)"].sum(), 1):<13} {round(all_curr_results["TP"].sum(), 2):<9} {round(all_curr_results["FP"].sum(), 2):<9} {round(all_curr_results["FN"].sum(), 2):<9} {get_precision(all_curr_results["TP"].sum(), all_curr_results["FP"].sum()):<10} {get_recall(all_curr_results["TP"].sum(), all_curr_results["FN"].sum()):<10} {get_f1(get_precision(all_curr_results["TP"].sum(), all_curr_results["FP"].sum()), get_recall(all_curr_results["TP"].sum(), all_curr_results["FN"].sum())):<10} {get_accuracy(all_curr_results["TP"].sum(), all_curr_results["FP"].sum(), all_curr_results["FN"].sum()):<10}'
            print(table_text)
    except FileNotFoundError:
        pass

    if not os.path.exists(f'{versioned_savefolder_name}/images'):
        os.makedirs(f'{versioned_savefolder_name}/images')
    if not os.path.exists(f'{major_versioned_savefolder_name}/bbs'):
        os.makedirs(f'{major_versioned_savefolder_name}/bbs')
    if not os.path.exists(f'{major_versioned_savefolder_name}/texts'):
        os.makedirs(f'{major_versioned_savefolder_name}/texts')

    for index, row in docids_df.iterrows():
        if row['docid'] in current_results['Document'].values:
            continue

        time0 = time.time()
        docid = row['docid']
        document_path = f'{document_folder}/{docid}.pdf'

        with open(document_path, 'rb') as file:
            pdf_bytes = file.read()

        dimensions_2 = evaluation_utils.get_pdf_dimensions_from_byte_file(pdf_bytes)

        true_labels = labels_df[labels_df['docid'] == docid]

        images, all_text, model_bbs, predicted_boxes, predicted_keyword, predicted_regex, image_dimensions = model_main.extended_model(
            pdf_bytes,
            save_bbs_path=major_versioned_savefolder_name + f'/bbs/{docid}',
            save_text_path=major_versioned_savefolder_name + f'/texts/{docid}'
        )

        ratio = image_dimensions[0] / dimensions_2[0]

        page_count = evaluation_utils.get_pdf_pagecount(document_path)

        images_true, dimensions_pdf = model_utils.convert_pdf_bytes_to_images(pdf_bytes)

        metrics_list = []

        true_boxes_doc = []

        # Loop through all pages
        for i in range(page_count):

            true_boxes_page = []

            try:
                predicted_boxes_page = predicted_boxes[i]
                true_boxes_page_df = true_labels[true_labels['sidetall'] == i + 1]

                for index_2, box_row in true_boxes_page_df.iterrows():
                    box = [box_row['height'], box_row['width'], box_row['x'], box_row['y']]
                    true_boxes_page.append(evaluation_utils.scale_bounding_box(box, ratio))

                matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(true_boxes_page, predicted_boxes_page)
                metrics_list.append(metrics)
            except:
                print(f"No boxes on page {i + 1}")
            true_boxes_doc.append(true_boxes_page)

        results = evaluation_utils.metrics_perdocument(metrics_list)

        # images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, predicted_boxes, true_boxes_doc)
        images_with_bbs = evaluation_utils.visualize_bounding_boxes_detailed(images_true, model_bbs, true_boxes_doc, predicted_keyword, predicted_regex, show=False)

        for i, img in enumerate(images_with_bbs):
            img.savefig(f'{versioned_savefolder_name}/images/{docid}_{i}.png')

        current_results = pd.concat([current_results, pd.DataFrame({
            'Document': docid,
            'Page': page_count,
            'Time(s)': round(time.time() - time0, 1),
            'TP': results['TP'],
            'FP': results['FP'],
            'FN': results['FN']
        }, index=[0])], ignore_index=True)
        current_results.to_csv(results_path, index=False)

        if index % header_printing_interval == 0:
            print(header_text)

        table_text = f'{index:<5} {round(time.time() - time0, 1):<8} {docid:<20} {page_count:<6} {results["TP"]:<3} {results["FP"]:<3} {results["FN"]:<3} {round(time.time() - total_time, 1):<13} {current_results["TP"].sum():<9} {current_results["FP"].sum():<9} {current_results["FN"].sum():<9} {get_precision(current_results["TP"].sum(), current_results["FP"].sum()):<10} {get_recall(current_results["TP"].sum(), current_results["FN"].sum()):<10} {get_f1(get_precision(current_results["TP"].sum(), current_results["FP"].sum()), get_recall(current_results["TP"].sum(), current_results["FN"].sum())):<10} {get_accuracy(current_results["TP"].sum(), current_results["FP"].sum(), current_results["FN"].sum()):<10}'
        print(table_text)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    current_model_version_number = "1.0"

    evaluate_model(
        current_model_version_number,
        # Input:
        "valideringssett/all_documents",
        "valideringssett/all_labels.csv",
        "valideringssett/all_docids.csv",

        #Output:
        "valideringssett/results"
    )

## RESULTS ALL DOCUMENTS
# Config: '--oem 1 --psm 11'
# With Ocr peronnummer
# Total TP: 686, Total FP: 9, Total FN: 823

# Precision: 0.9870503597122302
# Recall: 0.4546056991385023
# F1: 0.6225045372050816
# Accuracy: 0.4519104084321476


## RESULTS ELEKTRONISK TINGLYST
# Total TP: 162, Total FP: 0, Total FN: 0

##RESUULTS ALL DOCUMENTS()
# Config: '--oem 1 --psm 11'
# With Ocr peronnummer, boxsplitting and keywords
# Total TP: 909, Total FP: 56, Total FN: 599

# Precision: 0.9419689119170984
# Recall: 0.6027851458885941
# F1: 0.7351395066720583
# Accuracy: 0.5812020460358056

##RESUULTS ALL DOCUMENTS EASYOCR AFTER 444 DOCUMENTS
# With Ocr peronnummer, boxsplitting and keywords
# Total TP: 380, Total FP: 84, Total FN: 311

# Precision: 0.8189655172413793
# Recall: 0.5499276410998553
# F1: 0.6580086580086579
# Accuracy: 0.49032258064516127
