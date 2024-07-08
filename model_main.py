import model_utils

def main(pdf_file):
    """
    Extract text from a PDF file and blur out sensitive information.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of images.
    list: A list of bounding boxes.
    list: A list of texts.
    """

    images, dimension = model_utils.convert_pdf_bytes_to_images(pdf_file)

    predicted_boxes = []

    for i, image in enumerate(images):
        text, bounding_boxes = model_utils.extract_text_and_bb_from_image(image)

        tagged_matches = model_utils.find_matches(text)

        bbs = model_utils.get_boxes_to_blur(tagged_matches, bounding_boxes)

        predicted_boxes.append(bbs)

    return images, predicted_boxes, dimension


def inference(aar, id, embete):

    pdf_bytes = model_utils.download_pdf(aar, id, embete)

    images, predicted_boxes, dimension = main(pdf_bytes)

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
                "y": bb[1],
                "dimensions" : dimension
            })

    return images, predicted_boxes, json_responses, dimension