import model_utils

def model(pdf_file):
    """
    Extract text from a PDF file and blur out sensitive information.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of images.
    list: A list of bounding boxes.
    list: A list of texts.
    """

    images, dimensions = model_utils.convert_pdf_bytes_to_images(pdf_file)

    predicted_boxes = []

    for i, image in enumerate(images):
        text, bounding_boxes = model_utils.extract_text_and_bb_from_image(image)

        tagged_matches = model_utils.find_matches(text)

        bbs = model_utils.get_boxes_to_blur(tagged_matches, bounding_boxes)

        predicted_boxes.append(bbs)

    return images, predicted_boxes, dimensions


def main(docid):
    """
    Extract text from a PDF file and blur out sensitive information.

    Parameters:
    docid (str): the document ID.

    Returns:
    images (list): A list of images.
    predicted_boxes (list): A list of bounding boxes per image.
    json_responses (list): A list of json responses (dict).
    ratio (float): The ratio between the PDF and image dimensions.
    """

    pdf_bytes = model_utils.download_pdf(docid)

    pdf_dimensions = model_utils.get_pdf_dimensions_from_byte_file(pdf_bytes)

    images, predicted_boxes, image_dimensions = model(pdf_bytes)

    ratio = pdf_dimensions[0] / image_dimensions[0]

    predicted_boxes_scaled = model_utils.scale_and_pad_all_bounding_boxes(predicted_boxes, ratio)
    
    json_responses = []

    for page_num, bb_page in enumerate(predicted_boxes_scaled):
        for bb_index, bb in enumerate(bb_page):
            json_responses.append({
                "page": page_num+1,
                "height": bb[0],
                "width": bb[1],
                "x": bb[2],
                "y": bb[3]
            })

    return images, json_responses, ratio, predicted_boxes_scaled

if __name__ == '__main__':
    res = main('2023_62529_200')
    print(res)
