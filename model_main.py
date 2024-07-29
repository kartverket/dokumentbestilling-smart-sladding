import model_utils
import time
import pandas as pd

def model(pdf_file, run_tesseract=True, run_easyocr=True, run_keyword_search=True, languages=['no', 'da', 'en'], tess_config=r'--oem 1 --psm 11', num_indexes=3, num_closest=[6,12]):
    """
    Extract text from a PDF file and blur out sensitive information.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of images.
    list: A list of bounding boxes.
    list: A list of texts.
    """

    predicted_boxes = []

    elektronisk_tinglyst = model_utils.is_elektronisk_tinglyst(pdf_file)

    #Logger
    if elektronisk_tinglyst:
        print('Elektronisk tinglyst, skipping keyword detection')
    else:
        print('Not elektronisk tinglyst, running keyword detection')

    images, dimensions = model_utils.convert_pdf_bytes_to_images(pdf_file)

    for i, image in enumerate(images):

        bounding_boxes, text = model_utils.ocr(image, run_tesseract, run_easyocr, languages, tess_config, elektronisk_tinglyst)
        predicted_boxes_regex = model_utils.apply_regex_search(bounding_boxes, text)

        if not elektronisk_tinglyst:

            if run_keyword_search:

                predicted_boxes_keyword = model_utils.apply_keyword_search(bounding_boxes, num_indexes, num_closest)

                all_boxes = predicted_boxes_regex + predicted_boxes_keyword

                predicted_boxes.append(all_boxes)

        if elektronisk_tinglyst:
            predicted_boxes.append(predicted_boxes_regex)

    clean_predicted_boxes = model_utils.remove_duplicated_boxes(predicted_boxes)

    return clean_predicted_boxes, dimensions


def main(docid, base_url):
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
    pdf_bytes = model_utils.download_pdf(docid, base_url)

    pdf_dimensions = model_utils.get_pdf_dimensions_from_byte_file(pdf_bytes)

    predicted_boxes, image_dimensions = model(pdf_bytes)
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
                "y": bb[3],
                "type": "PERSONNUMMER",
                "ml_generated": "true",
            })

    return json_responses
    
if __name__ == '__main__':
    res = main('2023_62529_200',"https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok")
    print(res)