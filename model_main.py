import model_utils
import numpy as np

def main(pdf_path):

    images, dimensions = model_utils.convert_pdf_to_images(pdf_path)

    predicted_boxes = []
    texts = []

    for i, image in enumerate(images):
        text, bounding_boxes = model_utils.extract_text_and_bb_from_image(image)
        tagged_matches = model_utils.find_matches(text)
        bbs = model_utils.get_boxes_to_blur(tagged_matches, bounding_boxes)
        predicted_boxes.append(bbs)
        texts.append(text)

    return images, predicted_boxes, texts
