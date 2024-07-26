import evaluation_utils
import model_main
import pandas as pd
import os
import model_utils
import time
import json


def evaluate_model(folder_path, labels_path, savefolder_name, config = r'--oem 3 --psm 11', num_indexes = 3, num_closest_above = 3, num_closest_below = 7):
    """
    Evaluate the model on a set of documents.

    Parameters:
    folder_path (str): The path to the folder containing the documents.
    labels_path (str): The path to the csv file containing the labels.
    savefolder_name (str): The name of the folder where the results will be saved.
    model_function (function): The function that will be used to extract the text and bounding boxes from the images.

    Returns:
    total_results (list): A list of dictionaries containing the results per document.
    total_tp (int): The total number of true positives.
    total_fp (int): The total number of false positives.
    total_fn (int): The total number of false negatives.
    df_results (pd.DataFrame): A DataFrame containing the results per document.
    """

    base_url = "https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok"

    organized_labels_path = pd.read_csv(labels_path)

    total_results = []
    total_tp, total_fp, total_fn = 0,0,0

    docids = []
    tps = []
    fps = []
    fns = []

    #Loop through all docu
    for index, dokument in enumerate(os.listdir(folder_path)):
        pdf_path = folder_path + dokument
        #remove .pdf from the name
        docid = dokument[:-4]
        docids.append(docid)

        pdf_bytes = model_utils.download_pdf(docid, base_url)
        languages = ['no', 'en', 'da']
        images, all_text, model_bbs, predicted_boxes, predicted_keyword, predicted_regex, image_dimensions, keywords_dict = model_main.model(pdf_bytes, languages, config, num_indexes=num_indexes, num_closest_above=num_closest_above, num_closest_below=num_closest_below)

        for i, text in enumerate(all_text):
            with open(f'{savefolder_name}/{docid}_{i}.txt', 'w') as file:
                file.write(text)
        
        #Save keywords_and_ssn_found dictionary as .json file
        with open(f'{savefolder_name}/{docid}_keywords_and_ssn_found.json', 'w') as file:
            json.dump(keywords_dict, file, indent=4)

        images_true, true_boxes = evaluation_utils.get_images_and_bb_from_docid(organized_labels_path, docid, folder_path)


        #Get the true positives, false positives and false negatives
        metrics_list = []
        for i,j in zip(true_boxes, predicted_boxes):
            matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(i, j)
            metrics_list.append(metrics)


        results = evaluation_utils.metrics_perdocument(metrics_list)


        images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, model_bbs, true_boxes, predicted_keyword, predicted_regex, show=False)

        for i, img in enumerate(images_with_bbs):
            img.savefig(f'{savefolder_name}/{docid}_{i}.png')
    
        tps.append(results['TP'])
        fps.append(results['FP'])
        fns.append(results['FN'])
    
        total_tp += results['TP']
        total_fp += results['FP']
        total_fn += results['FN']

        print(total_tp, total_fp, total_fn)
        
        total_results.append(results)
        print(index)

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")

    df_results = pd.DataFrame({'docid': docids, 'TP': tps, 'FP': fps, 'FN': fns})
    df_results.to_csv(f'{savefolder_name}/results_per_doc.csv', index=False)

    return total_results, total_tp, total_fp, total_fn, df_results


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