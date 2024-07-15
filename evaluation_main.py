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

    organized_labels_path = pd.read_csv("../valideringssett/organized_data.csv")

    total_results = []
    total_tp, total_fp, total_fn = 0,0,0

    #Loop through all docu
    for index, dokument in enumerate(os.listdir(folder_path)):
        pdf_path = folder_path + dokument
        #remove .pdf from the name
        docid = dokument[:-4]

        images_true, true_boxes = evaluation_utils.get_images_and_bb_from_docid(organized_labels_path, docid)

        images, dimensions = evaluation_utils.convert_pdf_path_to_images(pdf_path)

        predicted_boxes = []

        total_text = ""

        time0 = time.time()
        for i, image in enumerate(images):
            text, bounding_boxes = model_utils.extract_text_and_bb_from_image(image, config)

            tagged_matches = model_utils.find_matches(text)

            bbs = model_utils.get_boxes_to_blur(tagged_matches, bounding_boxes)

            predicted_boxes.append(bbs)

            total_text += text
        time1 = time.time()
        print(f"OCR time for {docid}: {time1-time0}")
        
        metrics_list = []
        for i,j in zip(true_boxes, predicted_boxes):
            matched_boxes, unmatched_preds, metrics = evaluation_utils.match_bboxes(i, j)
            metrics_list.append(metrics)

        results = evaluation_utils.metrics_perdocument(metrics_list)

        images_with_bbs = evaluation_utils.visualize_bounding_boxes(images_true, true_boxes, predicted_boxes, show=False)
        
        if results['FP'] > 0 or results['FN'] > 0:
            for i, img in enumerate(images_with_bbs):
                img.savefig(f'../wrong_all/{docid}_{i}.png')
                
        if results['FP'] > 0:
            os.mkdir(f'../wrong_fp/{docid}/')
            for i, img in enumerate(images_with_bbs):
                img.savefig(f'../wrong_fp/{docid}/{i}.png')
                #Save textfile with the text
            with open(f"../wrong_fp/{docid}/{i}.txt", "w") as text_file:
                text_file.write(total_text)
            
        
        for i, img in enumerate(images_with_bbs):
            img.savefig(f'../resultater_all/{docid}_{i}.png')

        total_tp += results['TP']
        total_fp += results['FP']
        total_fn += results['FN']

        print(total_tp, total_fp, total_fn)
        
        total_results.append(results)
        print(index)

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")

    return total_results, total_tp, total_fp, total_fn

    