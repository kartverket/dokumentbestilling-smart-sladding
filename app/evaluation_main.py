import evaluation_utils
import model_main
import pandas as pd
import os
import model_utils
import time
import json

def evaluate_model(document_folder, labels_csv, docids_csv, savefolder_name):
    """
    Evaluate the model on a set of documents and save the images with bounding boxes(ground truth and predicted) and the text extracted from the documents.

    Parameters:
    document_folder (str): The path to the folder containing the documents.
    labels_csv (str): The path to the CSV file containing the labels. Has to contain only the accepted labels(The correct labels).
    docids_csv (str): The path to the CSV file containing the document IDs.
    savefolder_name (str): The path to the folder to save the results to.
    """


    #Load labels
    labels_df = pd.read_csv(labels_csv)
    labels_df['docid'] = labels_df['dokument_aar'].astype(str) + '_' + labels_df['dokument_nr'].astype(str) + '_' + labels_df['embete'].astype(str)

    #load document-ids
    docids_df = pd.read_csv(docids_csv)
    docids_df['docid'] = docids_df['dokument_aar'].astype(str) + '_' + docids_df['dokument_nr'].astype(str) + '_' + docids_df['embete'].astype(str)

    total_results = []
    total_tp, total_fp, total_fn = 0,0,0

    docids = []
    tps = []
    fps = []
    fns = []

    for index, row in docids_df.iterrows():

        print(index)

        docid = row['docid']
        print(docid)
        document_path = f'{document_folder}/{docid}.pdf'

        with open(document_path, 'rb') as file:
            pdf_bytes = file.read()

        dimensions_2 = evaluation_utils.get_pdf_dimensions_from_byte_file(pdf_bytes)

        true_labels = labels_df[labels_df['docid'] == docid]

        predicted_boxes, dimensions_model = model_main.model(pdf_bytes)

        images, all_text, model_bbs, predicted_boxes, predicted_keyword, predicted_regex, image_dimensions = extended_model(pdf_bytes)

        ratio = dimensions_model[0]/dimensions_2[0]

        page_count = evaluation_utils.get_pdf_pagecount(document_path)

        images_true, dimensions_pdf = model_utils.convert_pdf_bytes_to_images(pdf_bytes)

        metrics_list = []

        true_boxes_doc = []

        #Loop through all pages
        for i in range(page_count):

            true_boxes_page = []

            try:
                predicted_boxes_page = predicted_boxes[i]
                true_boxes_page_df = true_labels[true_labels['sidetall'] == i+1]

                for index_2, box_row in true_boxes_page_df.iterrows():
                    box = [box_row['height'], box_row['width'], box_row['x'], box_row['y']]
                    true_boxes_page.append(evaluation_utils.scale_bounding_box(box, ratio))

                matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(true_boxes_page, predicted_boxes_page)
                metrics_list.append(metrics)
            except:
                print(f"No boxes on page {i+1}")
            true_boxes_doc.append(true_boxes_page)

        results = evaluation_utils.metrics_perdocument(metrics_list)

        #images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, predicted_boxes, true_boxes_doc)
        images_with_bbs = evaluation_utils.visualize_bounding_boxes_detailed(images_true, model_bbs, true_boxes_doc, predicted_keyword, predicted_regex, show=False)

        for i, img in enumerate(images_with_bbs):
            img.savefig(f'{savefolder_name}/{docid}_{i}.png')

        for i, text in enumerate(all_text):
            with open(f'{savefolder_name}/{docid}_{i}.txt', 'w') as file:
                file.write(text)
    
        tps.append(results['TP'])
        fps.append(results['FP'])
        fns.append(results['FN'])
    
        total_tp += results['TP']
        total_fp += results['FP']
        total_fn += results['FN']

        print(f'True Positives on doc: {results["TP"]}, False Positives on doc: {results["FP"]}, False Negatives on doc: {results["FN"]}')
        print(f'Total True Positives: {total_tp}, Total False Positives: {total_fp}, Total False Negatives:  {total_fn}')
        
        total_results.append(results)

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")

    df_results = pd.DataFrame({'docid': docids, 'TP': tps, 'FP': fps, 'FN': fns})
    df_results.to_csv(f'{savefolder_name}/results_per_doc.csv', index=False)

    return total_results, total_tp, total_fp, total_fn, df_results


def extended_model(pdf_file, run_tesseract=True, run_easyocr=True, run_keyword_search=True, languages = ['no', 'da', 'en'], tess_config=r'--oem 1 --psm 11', num_indexes=3, num_closest=[6,12]):
    """
    Extract text from a PDF file and blur out sensitive information.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of images.
    list: A list of bounding boxes.
    list: A list of texts.
    """

    elektronisk_tinglyst = model_utils.is_elektronisk_tinglyst(pdf_file)

    #Logger
    if elektronisk_tinglyst:
        print('Elektronisk tinglyst, skipping keyword detection')
    else:
        print('Not elektronisk tinglyst, running keyword detection')

    images, dimensions = model_utils.convert_pdf_bytes_to_images(pdf_file)

    model_bbs = []
    all_text = []
    predicted_boxes = []
    all_predicted_bbs_keywords = []
    all_predicted_bbs_regex = []

    for i, image in enumerate(images):

        bounding_boxes, text = model_utils.ocr(image, run_tesseract, run_easyocr, languages, tess_config, elektronisk_tinglyst)
        predicted_boxes_regex = model_utils.apply_regex_search(bounding_boxes, text)

        model_bbs.append(model_utils.get_all_bbs(bounding_boxes))
        all_text.append(text)
        all_predicted_bbs_regex.append(predicted_boxes_regex)

        if not elektronisk_tinglyst:

            if run_keyword_search:

                predicted_boxes_keyword = model_utils.apply_keyword_search(bounding_boxes, num_indexes, num_closest)

                all_predicted_bbs_keywords.append(predicted_boxes_keyword)

                all_boxes = predicted_boxes_regex + predicted_boxes_keyword

                unique_bounding_boxes = model_utils.remove_duplicates(all_boxes)

                predicted_boxes.append(unique_bounding_boxes)
            
            else:
                predicted_boxes.append(predicted_boxes_regex)

        if elektronisk_tinglyst:
            predicted_boxes.append(predicted_boxes_regex)

    clean_predicted_boxes = model_utils.remove_overlapping_boxes(predicted_boxes)

    return images, all_text, model_bbs, clean_predicted_boxes, all_predicted_bbs_keywords, all_predicted_bbs_regex, dimensions


## RESULTS ALL DOCUMENTS
# Config: '--oem 1 --psm 11'
# With Ocr peronnummer
#Total TP: 686, Total FP: 9, Total FN: 823

#Precision: 0.9870503597122302
#Recall: 0.4546056991385023
#F1: 0.6225045372050816
#Accuracy: 0.4519104084321476


## RESULTS ELEKTRONISK TINGLYST
#Total TP: 162, Total FP: 0, Total FN: 0

##RESUULTS ALL DOCUMENTS()
# Config: '--oem 1 --psm 11'
# With Ocr peronnummer, boxsplitting and keywords
# Total TP: 909, Total FP: 56, Total FN: 599

#Precision: 0.9419689119170984
#Recall: 0.6027851458885941
#F1: 0.7351395066720583
#Accuracy: 0.5812020460358056

##RESUULTS ALL DOCUMENTS EASYOCR AFTER 444 DOCUMENTS
# With Ocr peronnummer, boxsplitting and keywords
# Total TP: 380, Total FP: 84, Total FN: 311

#Precision: 0.8189655172413793
#Recall: 0.5499276410998553
#F1: 0.6580086580086579
#Accuracy: 0.49032258064516127