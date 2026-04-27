import model_utils
import pandas as pd
import os
from typing import List, Dict

import hjemmel_search_utils
from url_utils import api_base_url

import logging


def process_pdf(pdf_file, run_tesseract=True, run_easyocr=True, run_keyword_search=True, run_hjemmel_search=True,
                languages=['no', 'da', 'en'], tess_config=r'--oem 1 --psm 11',
                num_indexes=3, num_closest=[6, 12], extended=False, save_bbs_path=None, save_text_path=None, only_first_page=False):
    """
    Process a PDF file to extract text and predict bounding boxes for sensitive information.

    Parameters:
    - pdf_file: The PDF file to process.
    - extended (bool): Flag to control the extent of outputs.

    Returns:
    - Depending on the extended flag, returns a tuple with various outputs.
    """

    elektronisk_tinglyst = model_utils.is_elektronisk_tinglyst(pdf_file)

    '''
    # Logger
    if elektronisk_tinglyst:
        logging.info('Elektronisk tinglyst, skipping keyword detection')
    else:
        logging.info('Not elektronisk tinglyst, running keyword detection')
    '''

    images, dimensions = model_utils.convert_pdf_bytes_to_images(pdf_file, only_first_page=only_first_page)

    predicted_boxes = []

    model_bbs = []
    all_text = []
    all_predicted_bbs_keywords = []
    all_predicted_bbs_regex = []
    all_predicted_bbs_hjemmel = []

    for i, image in enumerate(images):

        save_bbs_path_page = save_bbs_path + f'_{i}.csv' if save_bbs_path else None
        save_text_path_page = save_text_path + f'_{i}.txt' if save_text_path else None

        if (save_bbs_path_page is not None and save_text_path_page is not None and
                os.path.exists(save_bbs_path_page) and os.path.exists(save_text_path_page)):
            bounding_boxes = pd.read_csv(save_bbs_path_page, dtype={'text': str, 'type': str}, keep_default_na=False)
            with open(save_text_path_page, 'r') as file:
                text = file.read()
        else:
            bounding_boxes, text = model_utils.ocr(
                image, run_tesseract, run_easyocr, languages, tess_config, elektronisk_tinglyst)
            if save_bbs_path_page:
                bounding_boxes.to_csv(save_bbs_path_page, index=False)
            if save_text_path_page:
                with open(save_text_path_page, 'w') as file:
                    file.write(text)

        predicted_boxes_regex_tesseract = model_utils.apply_regex_search(
            bounding_boxes[bounding_boxes['type'] == 'tesseract'], text)
        predicted_boxes_regex_easyocr = model_utils.apply_regex_search(
            bounding_boxes[bounding_boxes['type'] == 'easyocr'], text)

        predicted_boxes_regex = predicted_boxes_regex_tesseract + predicted_boxes_regex_easyocr
        predicted_boxes_keyword = []
        predicted_boxes_hjemmel = []

        if extended:
            model_bbs.append(model_utils.get_all_bbs(bounding_boxes))
            all_text.append(text)
            all_predicted_bbs_regex.append(predicted_boxes_regex)

        if not elektronisk_tinglyst and run_keyword_search:
            predicted_boxes_keyword = model_utils.apply_keyword_search(
                bounding_boxes, num_indexes, num_closest)
            if extended:
                all_predicted_bbs_keywords.append(predicted_boxes_keyword)

        if not elektronisk_tinglyst and run_hjemmel_search:
            predicted_boxes_hjemmel = hjemmel_search_utils.apply_hjemmel_search(bounding_boxes)
            if extended:
                all_predicted_bbs_hjemmel.append(predicted_boxes_hjemmel)

        all_boxes = predicted_boxes_regex + predicted_boxes_keyword + predicted_boxes_hjemmel
        unique_bounding_boxes = model_utils.remove_duplicates(all_boxes)
        predicted_boxes.append(unique_bounding_boxes)

    clean_predicted_boxes = model_utils.remove_overlapping_boxes(predicted_boxes)

    if extended:
        return (images, all_text, model_bbs, clean_predicted_boxes,
                all_predicted_bbs_keywords, all_predicted_bbs_regex, all_predicted_bbs_hjemmel, dimensions)
    else:
        return clean_predicted_boxes, dimensions


def extended_model(pdf_file, run_tesseract=True, run_easyocr=True, run_keyword_search=True, run_hjemmel_search=True,
                   languages=['no', 'da', 'en'], tess_config=r'--oem 1 --psm 11',
                   num_indexes=3, num_closest=[6, 12], save_bbs_path=None, save_text_path=None, only_first_page=False):
    """
    Extended model that extracts text and additional information from a PDF file.
    """
    return process_pdf(pdf_file, run_tesseract, run_easyocr, run_keyword_search, run_hjemmel_search,
                       languages, tess_config, num_indexes, num_closest, extended=True, save_bbs_path=save_bbs_path, save_text_path=save_text_path,
                       only_first_page=only_first_page)


def model(pdf_file, run_tesseract=True, run_easyocr=True, run_keyword_search=True, run_hjemmel_search=True,
          languages=['no', 'da', 'en'], tess_config=r'--oem 1 --psm 11',
          num_indexes=3, num_closest=[6, 12]):
    """
    Basic model that extracts text and predicts bounding boxes from a PDF file.
    """
    return process_pdf(pdf_file, run_tesseract, run_easyocr, run_keyword_search, run_hjemmel_search,
                       languages, tess_config, num_indexes, num_closest, extended=False)


def run_model_on_pdf_bytes(pdf_bytes: bytes) -> List[Dict]:
    """
    Process PDF bytes to extract and scale bounding boxes, then format them as JSON responses.

    Parameters:
        pdf_bytes (bytes): The PDF file as bytes.

    Returns:
        List[Dict]: A list of JSON responses containing bounding box information.
    """
    # Extract predicted boxes and image dimensions using the model
    predicted_boxes, image_dimensions = model(pdf_bytes)

    # Retrieve PDF dimensions
    pdf_dimensions = model_utils.get_pdf_dimensions_from_byte_file(pdf_bytes)
    if not pdf_dimensions:
        raise ValueError("Failed to retrieve PDF dimensions.")

    ratio = pdf_dimensions[0] / image_dimensions[0]

    # Scale and pad the bounding boxes based on the ratio
    predicted_boxes_scaled = model_utils.scale_and_pad_all_bounding_boxes(predicted_boxes, ratio)

    # Generate JSON responses using list comprehension for efficiency
    json_responses = [
        {
            "page": page_num + 1,
            "height": bb[0],
            "width": bb[1],
            "x": bb[2],
            "y": bb[3],
        }
        for page_num, bb_page in enumerate(predicted_boxes_scaled)
        for bb in bb_page
    ]

    return json_responses


def get_pdf_bytes(document_url: str) -> bytes:
    """
    Retrieve PDF bytes from a local file path or a remote URL.

    Parameters:
        document_url (str): The URL or local path to the PDF document.

    Returns:
        bytes: The PDF file as bytes.

    Raises:
        ValueError: If the document URL is invalid or the download fails.
    """
    try:
        if document_url.lower().startswith('http'):
            pdf_bytes = model_utils.download_pdf(document_url)
            if not pdf_bytes:
                raise ValueError(f"Failed to download PDF from URL: {document_url}")
        else:
            with open(document_url, 'rb') as file:
                pdf_bytes = file.read()
        return pdf_bytes
    except Exception as e:
        raise ValueError(f"Error retrieving PDF bytes: {e}")


def main(document_url: str) -> List[Dict]:
    """
    Extract bounding box information from a PDF file and format it as JSON responses.

    Parameters:
        document_url (str): The URL or local path to the PDF document.

    Returns:
        List[Dict]: A list of JSON responses containing bounding box information.
    """
    pdf_bytes = get_pdf_bytes(document_url)
    json_responses = run_model_on_pdf_bytes(pdf_bytes)
    return json_responses


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    docid = '2023_72893_200'
    res = main(f'{api_base_url()}intern/pantebok/gjenpart/{docid}?attestering=false')
    logging.info(res)
