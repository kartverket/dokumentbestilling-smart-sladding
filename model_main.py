import model_utils
import time
import pandas as pd

def model(pdf_file, languages, config, num_indexes, num_closest_above, num_closest_below):
    """
    Extract text from a PDF file and blur out sensitive information.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    list: A list of images.
    list: A list of bounding boxes.
    list: A list of texts.
    """

    elektronisk_tinglyst = model_utils.is_elektronisk_tinglyst(pdf_file)

    #Logger
    if elektronisk_tinglyst:
        print('Elektronisk tinglyst, skipping keyword detection')
    else:
        print('Not elektronisk tinglyst, running keyword detection')

    images, dimensions = model_utils.convert_pdf_bytes_to_images(pdf_file)

    model_bbs = []
    all_text = []
    predicted_boxes = []
    predicted_bbs_keywords = []
    predicted_bbs_regex = []

    keywords_dict = {}

    for i, image in enumerate(images):
        text_tess, bounding_boxes_tess = model_utils.apply_tesseractocr(image, languages, config, elektronisk_tinglyst)
        text_easy, bounding_boxes_easy = model_utils.apply_easyocr(image, languages, config, elektronisk_tinglyst)

        text = text_tess + ' ' + text_easy

        bounding_boxes = pd.concat([bounding_boxes_tess, bounding_boxes_easy], ignore_index=True)

        model_bbs.append(model_utils.get_all_bbs(bounding_boxes))
        all_text.append(text)

        tagged_matches = model_utils.find_matches(text)

        bbs = model_utils.get_boxes_to_blur(tagged_matches, bounding_boxes)
        predicted_bbs_regex.append(bbs)

        if not elektronisk_tinglyst:

            keyword_boxes, keywords_and_ssn_found = model_utils.get_bbs_from_keywords(bounding_boxes, num_indexes = num_indexes, num_closest_above = num_closest_above, num_closest_below=num_closest_below)
            predicted_bbs_keywords.append(keyword_boxes)

            keywords_dict[i] = keywords_and_ssn_found


            all_boxes = bbs + keyword_boxes

            bounding_boxes_tuples = [tuple(box) for box in all_boxes]
            unique_bounding_boxes_tuples = set(bounding_boxes_tuples)
            unique_bounding_boxes = [list(box) for box in unique_bounding_boxes_tuples]

            predicted_boxes.append(unique_bounding_boxes)

        if elektronisk_tinglyst:
            predicted_boxes.append(bbs)

    clean_predicted_boxes = model_utils.remove_duplicated_boxes(predicted_boxes)

    return images, all_text, model_bbs, clean_predicted_boxes, predicted_bbs_keywords, predicted_bbs_regex, dimensions, keywords_dict


def main(docid, base_url, languages = ['no', 'en', 'da'], config = r'--oem 1 --psm 11', num_indexes = 3, num_closest_above = 3, num_closest_below = 7):
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

    time2 = time.time()
    images, all_text, model_bbs, predicted_boxes, predicted_keyword, predicted_regex, image_dimensions, keywords_and_ssn_found = model(pdf_bytes, languages, config, num_indexes=num_indexes, num_closest_above=num_closest_above, num_closest_below=num_closest_below)
    time3 = time.time()
    print(f"Ran model function for document {docid} in {time3-time2} seconds.")

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

    return json_responses

if __name__ == '__main__':
    res = main('2023_62529_200',"https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok", languages = ['no', 'en', 'da'], config = r'--oem 1 --psm 11', num_indexes=3, num_closest_above=6, num_closest_below=12)
    print(res)