import evaluation_utils
import model_main
import pandas as pd
import os
import model_utils
import time


def evaluate_model(folder_path, config = r'--oem 3 --psm 11'):
    """
    Evaluate the model on a set of documents.

    Parameters:
    folder_path (str): The path to the folder containing the documents.

    Returns:
    total_results (list): A list of dictionaries containing the results per document.
    total_tp (int): The total number of true positives.
    total_fp (int): The total number of false positives.
    total_fn (int): The total number of false negatives.
    """

    base_url = "https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok"

    organized_labels_path = pd.read_csv("valideringssett/organized_labels.csv")

    total_results = []
    total_tp, total_fp, total_fn = 0,0,0

    #Loop through all docu
    for index, dokument in enumerate(os.listdir(folder_path)):
        pdf_path = folder_path + dokument
        #remove .pdf from the name
        docid = dokument[:-4]
        print(docid)

        json_responses, predicted_boxes, images = model_main.main(docid, base_url)

        images_true, true_boxes = evaluation_utils.get_images_and_bb_from_docid(organized_labels_path, docid, folder_path)

        #Get the true positives, false positives and false negatives
        metrics_list = []
        for i,j in zip(true_boxes, predicted_boxes):
            matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(i, j)
            metrics_list.append(metrics)

        results = evaluation_utils.metrics_perdocument(metrics_list)


        images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, true_boxes, predicted_boxes, show=False)
        for i, img in enumerate(images_with_bbs):
            img.savefig(f'resultater_all/{docid}_{i}.png')
    
        total_tp += results['TP']
        total_fp += results['FP']
        total_fn += results['FN']

        print(total_tp, total_fp, total_fn)
        
        total_results.append(results)
        print(index)

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")

    return total_results, total_tp, total_fp, total_fn