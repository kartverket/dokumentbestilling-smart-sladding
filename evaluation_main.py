import evaluation_utils
import model_main
import pandas as pd
import os

def evaluate_model(folder_path):

    organized_labels_path = pd.read_csv("../valideringssett/organized_data.csv")

    total_results = []
    total_tp, total_fp, total_fn = 0,0,0
    #Loop through all docu
    for index, dokument in enumerate(os.listdir(folder_path)):
        pdf_path = folder_path + dokument
        #remove .pdf from the name
        doc_id = dokument[:-4]

        images_true, true_boxes = evaluation_utils.get_images_and_bb_from_docid(organized_labels_path, doc_id)
        images_pred, predicted_boxes, texts = model_main.main(pdf_path)

        metrics_list = []
        for i,j in zip(true_boxes, predicted_boxes):
            matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(i, j)
            metrics_list.append(metrics)

        results = evaluation_utils.metrics_perdocument(metrics_list)
        if results['FP'] > 0 or results['FN'] > 0:
            images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, true_boxes, predicted_boxes, show=False)
            for i, img in enumerate(images_with_bbs):
                img.savefig(f'../wrong_labels_all/{doc_id}_{i}.png')
        

        images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, true_boxes, predicted_boxes, show=False)
        for i, img in enumerate(images_with_bbs):
            img.savefig(f'../resultater_all/{doc_id}_{i}.png')

        total_tp += results['TP']
        total_fp += results['FP']
        total_fn += results['FN']
        
        total_results.append(results)
        print(index)

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")

    return total_results, total_tp, total_fp, total_fn

    