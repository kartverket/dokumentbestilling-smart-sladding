import evaluation_utils
import model_main
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageDraw
import matplotlib.patches as patches
import numpy as np
import os

def evaluate_model(folder_path):

    organized_labels_path = pd.read_csv("valideringssett/organized_data.csv")

    total_results = []
    total_tp, total_fp, total_fn = 0,0,0
    #Loop through all docu
    for dokument in os.listdir(folder_path):
        pdf_path = folder_path + dokument
        #remove .pdf from the name
        doc_id = dokument[:-4]
        images_true, true_boxes = evaluation_utils.get_images_and_bb_from_docid(organized_labels_path, doc_id)
        images_pred, predicted_boxes = model_main.main(pdf_path)
        
        metrics_list = []
        for i,j in zip(true_boxes, predicted_boxes):
            matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(i, j)
            metrics_list.append(metrics)

        results = evaluation_utils.metrics_perdocument(metrics_list)
        total_tp += results['TP']
        total_fp += results['FP']
        total_fn += results['FN']
        
        total_results.append(results)

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")

    return total_results, total_tp, total_fp, total_fn


folder_path = 'valideringssett/dokumenter/'

doc_id = "2023_73325_200"

evaluation_utils.test_and_visualize_doc(doc_id)

    