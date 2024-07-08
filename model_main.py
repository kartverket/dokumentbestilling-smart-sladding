import model_utils
import numpy as np
import validation_set
import evaluation_utils
import model_utils

def main(pdf_path, path=True):
    
    if path:
        images, dimensions = model_utils.convert_pdf_path_to_images(pdf_path)
    else:
        images, dimensions = model_utils.convert_pdf_bytes_to_images(pdf_path)

    predicted_boxes = []
    texts = []

    for i, image in enumerate(images):
        text, bounding_boxes = model_utils.extract_text_and_bb_from_image(image)

        tagged_matches = model_utils.find_matches(text)

        bbs = model_utils.get_boxes_to_blur(tagged_matches, bounding_boxes)

        predicted_boxes.append(bbs)
        
        texts.append(text)

    return images, predicted_boxes, texts


def inference(aar, id, embete):

    pdf_bytes = validation_set.download_pdf(aar, id, embete)

    dimensions = evaluation_utils.get_pdf_dimensions_from_byte_file(pdf_bytes)

    images, predicted_boxes, texts = main(pdf_bytes, path=False)

    ratio = np.round(dimensions[0]/images[0].size[0], 6)

    predicted_boxes = model_utils.scale_all_bounding_boxes(predicted_boxes, ratio)

    json_responses = []

    for page_num, bb_page in enumerate(predicted_boxes):
        for bb_index, bb in enumerate(bb_page):
            json_responses.append({
                "dokument_aar": aar,
                "dokument_nr": id,
                "embete" : embete, 
                "sidetall": page_num,
                "index": bb_index,
                "type": "PERSONNUMMER",
                "ml_generated": True,
                "ml_status": None,
                "height": bb[3],
                "width": bb[2],
                "x": bb[0],
                "y": bb[1]
            })

    return images, predicted_boxes, json_responses, dimensions, ratio