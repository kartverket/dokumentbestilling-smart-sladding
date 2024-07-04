import evaluation_utils
import model_main
import pandas as pd


organized_labels_path = pd.read_csv("valideringssett/organized_data.csv")

#for index, doc in range(len(organized_labels_path)):

images, bbs = get_images_and_bb_from_index(organized_labels_path, 1)

visualize_bounding_boxes(organized_labels_path, 1)

for image, true_boxes in zip(images, bbs):
    #Model prediction boxes
    predicted_boxes = main(image)
    print(true_boxes)
    print(predicted_boxes)


    """ matched_boxes, unmatched_predictions, metrics = emu.match_bboxes(true_boxes, predicted_boxes)
    print(metrics) """

    