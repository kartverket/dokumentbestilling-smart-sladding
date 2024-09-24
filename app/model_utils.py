import re
import pytesseract
from pdf2image import convert_from_bytes
import requests
import fitz
import Levenshtein
from datetime import datetime
import numpy as np
import pandas as pd
import easyocr
from io import BytesIO
import torch
from torchvision.ops import box_iou
import copy as cp


#List with keywords and their corresponding allowed Levenshtein distance
keywords = [('personnr', 2), 
            ('pers nr', 1), 
            ('persnr', 1), 
            ('pnr', 0),
            ('fnr', 0),
            ('fødselsnr', 2), 
            ('fodselsnr', 2), 
            ('personnummer', 3), 
            ('fødselsnummer', 3), 
            ('fodselsnummer', 3), 
            ('fpnr', 1), 
            ('fødsnr', 1), 
            ('fodsnr', 1), 
            ('identifikasjonsnummer', 3),
            ('fnrorgnr', 1),
            ('fødselsnrorganisasjonsnr', 4),
            ('fødselsnrorgnr', 3),
            ('fødselsorganisasjonsnummer', 4),
            ('identifikasjonsnummer', 3),
            ('fødselsnrforetaksnr', 3),
            ('underorganisasjonsnrfødselsnr', 4)]


def download_pdf(docid, base_url):
    """
    Download a PDF file from a URL.

    Parameters:
    docid (str): The document ID on form "aar_id_embete".
    base_url (str): The base URL of the API.

    Returns:
    bytes: The content of the PDF file.
    """

    url = f"{base_url}/{docid}?attestering=false"
    response = requests.get(url)

    if response.status_code == 200:
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)

    return response.content


def convert_pdf_bytes_to_images(pdf_bytes):
    """
    Convert a PDF file to a list of images.

    Parameters:
    pdf_bytes (bytes): The PDF file as bytes.

    Returns:
    list: A list of images.
    tuple: A tuple containing the width and height of the images.
    """

    images = convert_from_bytes(pdf_bytes)

    width, height = images[0].size
    dimensions = (width, height)

    return images, dimensions


def pil_to_cv2(image):
    """
    Convert a PIL Image to an OpenCV image.

    Parameters:
    image (PIL.Image): The image to convert.

    Returns:
    np.array: The OpenCV image.
    """
    
    # Convert PIL Image to RGB
    image = image.convert('RGB')
    # Convert to numpy array
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def is_elektronisk_tinglyst(pdf_bytes):
    """
    Check if a PDF file is electronically registered.

    Parameters:
    pdf_bytes (bytes): The PDF file content in bytes.

    Returns:
    bool: True if the PDF file is electronically registered, False otherwise.
    """
     
    pdf_stream = BytesIO(pdf_bytes)

    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

    # Get and print the metadata
    metadata = pdf_document.metadata
    if metadata['title'] == 'Dokument til signering':
        return True
    else:
        return False
    

def get_pdf_dimensions_from_byte_file(pdf_bytes):
    """
    Get the dimensions of each page in a PDF file.

    Parameters:
    pdf_bytes (bytes): The PDF file content in bytes.

    Returns:
    list: A list of tuples where each tuple contains the width and height of a page.
    """
    # Open the PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    page = pdf_document.load_page(0)
    rect = page.rect
    return (rect.width, rect.height)


def remove_special_characters(text):
    """
    Remove special characters from a text.

    Parameters:
    text (str): The text to process.

    Returns:
    str: The text without special characters.
    """
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def format_bb_coordinates(bb):
    """
    Format bounding box coordinates to the format used by the model.

    Parameters:
    bb (list): A list of bounding box on form [(x, y), (x, y), (x, y), (x, y)].

    Returns:
    parameters: Formatted bounding box on form: w, h, x, y.
    """

    h = bb[3][1] - bb[0][1]
    w = bb[2][0] - bb[3][0]
    x = bb[0][0]
    y = bb[0][1]
    
    return h, w, x, y


def format_easyocr_result_to_df(result):
    """
    Format the result to a DataFrame.

    Parameters:
    result (list): A list containing the results.

    Returns:
    pd.DataFrame: The result as a DataFrame.
    """

    data_df = pd.DataFrame(columns=['left', 'top', 'width', 'height', 'text'])
    text = ""

    for line in result:
        h, w, x, y = format_bb_coordinates(line[0])
        new_data = pd.DataFrame([
        {'left': x, 'top': y, 'height': h, 'width': w, 'text': line[1]}
        ])
        data_df = pd.concat([data_df, new_data], ignore_index=True)

        text += line[1] + " "


    return data_df, text


def apply_tesseractocr(image, config = r'--oem 1 --psm 11', elektronisk_tinglyst = False):
    """
    Extract text and bounding boxes from an image.

    Parameters:
    image (PIL.Image?): The image to process.
    languages (list): A list of languages to use.
    config (str): The configuration string.

    Returns:
    str: The extracted text.
    pd.DataFrame: A DataFrame containing the bounding boxes.
    """

    text = pytesseract.image_to_string(image, config=config)
    text = text.replace('\n', ' ')
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=config)
    bounding_boxes = data[['left', 'top', 'width', 'height', 'text']]
    bounding_boxes = bounding_boxes.dropna()
    
    if not elektronisk_tinglyst:
        # Drop alle special characters
        bounding_boxes['text'] = bounding_boxes['text'].apply(remove_special_characters)
        text = remove_special_characters(text)
    return text, bounding_boxes


def apply_easyocr(image, languages = ['no', 'en', 'da'], elektronisk_tinglyst = False):
    """
    Apply EasyOCR to an image.

    Parameters:
    image (PIL.image): The image to process.
    languages (list): A list of languages to use.
    config (str): The configuration string.

    Returns:
    pd.DataFrame: A DataFrame containing the result.
    str: The extracted text.
    """
    image = pil_to_cv2(image)

    reader = easyocr.Reader(languages, model_storage_directory='../tmp/.EasyOCR/model', user_network_directory='../tmp/.EasyOCR/user_network')

    result = reader.readtext(image, width_ths = 0.01)

    result_df, text = format_easyocr_result_to_df(result)

    text = text.replace('\n', ' ')

    result_df = result_df.dropna()
    
    if not elektronisk_tinglyst:
        #drop alle special characters
        result_df['text'] = result_df['text'].apply(remove_special_characters)
        text = remove_special_characters(text)
    return text, result_df


def get_all_bbs(df):
    """
    Get all bounding boxes from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the bounding boxes.

    Returns:
    list: A list of bounding boxes.
    """

    all_bbs = []
    for iter, row in df.iterrows():
        bb = [row['height'], row['width'], row['left'], row['top']]
        all_bbs.append(bb)

    return all_bbs

def check_control_digits(fnr):
    """
    Validate the control digits of a Norwegian fødselsnummer or D-number.

    Parameters:
    fnr (str): The fødselsnummer or D-number to validate (11-digit string).

    Returns:
    bool: True if valid, False otherwise.
    """
    weights_first = [3, 7, 6, 1, 8, 9, 4, 5, 2]
    weights_second = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

    def calculate_control_digit(number, weights):
        total = sum(int(n) * w for n, w in zip(number, weights))
        remainder = total % 11
        if remainder == 0:
            return '0'
        elif remainder == 1:
            return None  # Invalid control digit
        else:
            return str(11 - remainder)

    first_control = calculate_control_digit(fnr[:9], weights_first)
    second_control = calculate_control_digit(fnr[:10], weights_second)

    return first_control == fnr[9] and second_control == fnr[10]

def find_matches(text):
    """
    Find and validate Norwegian fødselsnummer and D-numbers in a text, handling various formats.

    Parameters:
    text (str): The text to search for matches.

    Returns:
    list: A list of matches with corresponding tag ('fødselsnummer' or 'D-number') and index.
    """
    # Regular expression to match numbers with optional separators
    pattern = re.compile(
        r'\b'  # Word boundary
        r'(?:(?:\d{1,2}[\s\.\-\/]*){5}\d{1,5})'  # Matches digit groupings with optional separators
        r'\b'
    )
    tagged_matches = []
    index = 0

    matches = pattern.findall(text)
    for match in matches:
        # Remove all non-digit characters to get continuous digits
        fnr = re.sub(r'\D', '', match)

        if len(fnr) != 11 or not fnr.isdigit():
            continue  # Skip invalid matches

        day_str = fnr[:2]
        month_str = fnr[2:4]
        year_str = fnr[4:6]

        # Check if it's a D-number (day between 41 and 71)
        day = int(day_str)
        is_d_number = False
        if 41 <= day <= 71:
            day -= 40
            is_d_number = True

        # Build the birth date string
        birth_date_str = f"{day:02d}{month_str}{year_str}"

        # Attempt to parse the birth date
        try:
            datetime.strptime(birth_date_str, '%d%m%y')
        except ValueError:
            continue  # Invalid date, skip this match

        # Validate control digits
        if check_control_digits(fnr):
            tag = 'dnummer' if is_d_number else 'personnummer'
            tagged_matches.append([match, tag, index])
            index += 1

    return tagged_matches


def apply_regex_search(bounding_boxes, text):
    """
    Get bounding boxes for matches found in a text.

    Parameters:
    tagged_matches (list): A list of matches with corresponding tag and index.
    bounding_boxes (pd.DataFrame): A DataFrame containing the bounding boxes.

    Returns:
    list: A list of bounding boxes for the matches.
    """
    tagged_matches = find_matches(text)

    matches_list = []

    for i in tagged_matches:
        splitted = False
        sep_matches = re.split(r'\s+', i[0])

        if len(sep_matches[-1]) == 5:
            splitted = True
            matches_list.append([sep_matches[-1], i[1], i[2], splitted])
        
        else:
            matches_list.append([i[0], i[1], i[2], splitted])

    bbs = []
    for match in matches_list:
        pattern = re.compile(re.escape(match[0]), re.IGNORECASE)
        # Initialize a list to collect bounding boxes for each match
        match_bbs = []

        for index, row in bounding_boxes.iterrows():
            if pattern.search(row['text']):
                if match[3]:
                    loc = [row['height'], row['width'], row['left'], row['top']]
                else:
                    loc = [row['height'], 0.45*row['width'], row['left'] + 0.55*row['width'], row['top']]
                match_bbs.append(loc)
        
        # Extend the main bounding boxes list with all matches found for this pattern
        bbs.extend(match_bbs)

    bbs_clean = []
    for bb in bbs:
        if bb not in bbs_clean:
            bbs_clean.append(bb)

    return bbs_clean


def scale_and_pad_all_bounding_boxes(bounding_boxes, ratio, padding_factor = 0.2):
    """
    Scales all bounding boxes in a list by a given ratio.

    Parameters:
    bounding_boxes (list): A list of bounding boxes.
    ratio (float): The scaling ratio.
    padding_factor (float): The padding factor.

    Returns:
    list: A new list of bounding boxes scaled by the given ratio.
    """

    scaled_boxes = []

    for page in bounding_boxes:
        page_boxes = []
        for bbox in page:
            if bbox:
                height, width, x, y = bbox
                padding = height * padding_factor
                page_boxes.append([height * ratio + padding, width * ratio + padding, x * ratio - (padding / 2), y * ratio - (padding / 2)])
        scaled_boxes.append(page_boxes)

    return scaled_boxes

def apply_keyword_search(bounding_boxes, num_indexes=3, num_closest=[3,7]):
    """
    Apply keyword search to a DataFrame of bounding boxes.

    Parameters:
    bounding_boxes (pd.DataFrame): The DataFrame containing the bounding boxes.
    num_indexes (int): The number of indexes to search for.
    num_closest (list): The number of closest bounding boxes to find above and below the keyword.

    Returns:
    list: A list of bounding boxes.
    """

    predicted_boxes_keyword = get_bbs_from_keywords(bounding_boxes, num_indexes = num_indexes, num_closest_above = num_closest[0], num_closest_below=num_closest[1])

    return predicted_boxes_keyword


def find_closest_bounding_boxes(df, search_word, num_closest=10):
    """
    Find the closest bounding boxes to a word in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the bounding boxes.
    search_word (str): The word to search for.
    num_closest (int): The number of closest bounding boxes to find.

    Returns:
    pd.DataFrame: A DataFrame containing the closest bounding boxes.
    """

    # Search for all instances of the word in the DataFrame
    word_rows = df[df['text'] == search_word]

    if word_rows.empty:
        return False

    # Create a list to hold the closest bounding boxes for each instance of the search word
    all_closest_boxes = []

    # Calculate distances and find closest bounding boxes
    for _, word_row in word_rows.iterrows():
        # Get the bounding box coordinates of the found word
        word_bbox = word_row[['left', 'top', 'width', 'height']].values
        word_center = (word_bbox[0] + word_bbox[2] / 2, word_bbox[1] + word_bbox[3] / 2)

        # Calculate the Euclidean distance from the found word's center to all other bounding boxes' centers
        def calculate_distance(row):
            bbox_center = (row['left'] + row['width'] / 2, row['top'] + row['height'] / 2)
            return np.sqrt((word_center[0] - bbox_center[0])**2 + (word_center[1] - bbox_center[1])**2)

        df['distance'] = df.apply(calculate_distance, axis=1)

        # Sort by distance and select the five closest bounding boxes (excluding the word itself)
        closest_boxes = df[df['text'] != search_word].sort_values(by='distance').head(num_closest)

        # Append the closest boxes for this instance to the list
        all_closest_boxes.append(closest_boxes)

    # Concatenate all closest boxes DataFrames into one DataFrame
    closest_boxes_df = pd.concat(all_closest_boxes).drop_duplicates().reset_index(drop=True)

    return closest_boxes_df

def find_closest_bounding_boxes_constrained(df, search_word, num_closest_above=3, num_closest_below=7):
    """
    Find the closest bounding boxes to a word in a DataFrame, divided by above and below.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the bounding boxes.
    search_word (str): The word to search for.
    num_closest_above (int): The number of closest bounding boxes to find above the word.
    num_closest_below (int): The number of closest bounding boxes to find below the word.

    Returns:
    pd.DataFrame: A DataFrame containing the closest bounding boxes.
    """

    # Search for all instances of the word in the DataFrame
    word_rows = df[df['text'] == search_word]

    if word_rows.empty:
        return False

    # Create a list to hold the closest bounding boxes for each instance of the search word
    all_closest_boxes = []

    # Calculate distances and find closest bounding boxes
    for _, word_row in word_rows.iterrows():
        # Get the bounding box coordinates of the found word
        word_bbox = word_row[['left', 'top', 'width', 'height']].values
        word_center = (word_bbox[0] + word_bbox[2] / 2, word_bbox[1] + word_bbox[3] / 2)

        # Calculate the Euclidean distance from the found word's center to all other bounding boxes' centers
        def calculate_distance(row):
            bbox_center = (row['left'] + row['width'] / 2, row['top'] + row['height'] / 2)
            return np.sqrt((word_center[0] - bbox_center[0])**2 + (word_center[1] - bbox_center[1])**2)

        df['distance'] = df.apply(calculate_distance, axis=1)

        # Filter the boxes above and below
        above_boxes = df[(df['top'] < word_row['top']) & (df['text'] != search_word)]
        below_boxes = df[(df['top'] >= word_row['top']) & (df['text'] != search_word)]

        # Sort by distance and select the closest bounding boxes for above and below
        closest_above_boxes = above_boxes.sort_values(by='distance').head(num_closest_above)
        closest_below_boxes = below_boxes.sort_values(by='distance').head(num_closest_below)

        # Append the closest boxes for this instance to the list
        all_closest_boxes.append(closest_above_boxes)
        all_closest_boxes.append(closest_below_boxes)

    # Concatenate all closest boxes DataFrames into one DataFrame
    closest_boxes_df = pd.concat(all_closest_boxes).drop_duplicates().reset_index(drop=True)

    return closest_boxes_df


def get_levenshtein_distance(s1, s2):
    """
    Get the Levenshtein distance between two strings.

    Parameters:
    s1 (str): The first string.
    s2 (str): The second string.

    Returns:
    int: The Levenshtein distance between the two strings.
    """
    return Levenshtein.distance(s1, s2)


def remove_duplicates(bounding_boxes):
    """
    Get unique bounding boxes.
    Parameters:
    bounding_boxes (list): A list of bounding boxes.
    Returns:
    list: A list of unique bounding boxes.
    """
    bounding_boxes_tuples = [tuple(box) for box in bounding_boxes]
    unique_bounding_boxes_tuples = set(bounding_boxes_tuples)
    unique_bounding_boxes = [list(box) for box in unique_bounding_boxes_tuples]

    return unique_bounding_boxes


def can_be_ssn(s):
    """
    Check if a string can be converted to an integer or represents the last five digits of a number.

    Parameters:
    s (str): The string to check.

    Returns:
    str: 'whole_number' if the string can be a whole number, 'last_five' if the string is the last five digits of a number, False otherwise.
    """

    # Remove all whitespace in the string
    s = s.replace(" ", "")

    # Count the number of digits in the string
    int_count = 0
    for char in s:
        if char.isdigit():
            int_count += 1

    if s.startswith('00') or s.endswith('000'):
        return False

    # Return 'last_five' if the string is 4 or 5 digits long and contains more than 2 digits
    if len(s) == 5 and int_count > 3:
        return 'last_five'
    
    # Return 'whole_number' if the string is 10 or 11 digits long and contains more than 9 digits
    if len(s) > 9 and len(s) < 14 and int_count > 9: 
        return 'whole_number'
        
    return False
    

def get_bbs_from_keywords(bounding_boxes, num_indexes=3, num_closest_above=3, num_closest_below=7):
    """
    Get bounding boxes from keywords.

    Parameters:
    bounding_boxes (pd.DataFrame): The DataFrame containing the bounding boxes.

    Returns:
    list: A list of bounding boxes.
    """

    bounding_boxes = bounding_boxes.reset_index()
    
    indexes = []
    predicted_boxes = []
    for keyword in keywords:
        for row in bounding_boxes.iterrows():
            if get_levenshtein_distance(keyword[0], row[1]['text'].lower()) < keyword[1]+1:
                indexes.append([row[0], keyword[0]])

    for index, keyword_ in indexes:

        for next in range(index, index+num_indexes):
            if next < len(bounding_boxes):
                check  = can_be_ssn(bounding_boxes.iloc[next]['text'])
                if check == 'last_five':
                    row = bounding_boxes.iloc[next]
                    predicted_boxes.append([row['height'], row['width'], row['left'], row['top']])

                if check == 'whole_number':
                    row = bounding_boxes.iloc[next]
                    predicted_boxes.append([row['height'], 0.45*row['width'], row['left'] + 0.55*row['width'], row['top']])


        closest_bbs = find_closest_bounding_boxes_constrained(bounding_boxes, bounding_boxes.iloc[index]['text'], num_closest_above=num_closest_above, num_closest_below=num_closest_below)

        for row in closest_bbs.iterrows():
            check  = can_be_ssn(row[1]['text'])
            if check == 'last_five':
                predicted_boxes.append([row[1]['height'], row[1]['width'], row[1]['left'], row[1]['top']])

            if check == 'whole_number':
                predicted_boxes.append([row[1]['height'], 0.45*row[1]['width'], row[1]['left'] + 0.55*row[1]['width'], row[1]['top']])
                
            
    return predicted_boxes


def format_box_to_iou(box):
    """
    Formats a box on form (height, width, left, top) to the format required by the box_iou function in torchvision (x1, y1, x2, y2).
    Parameters:
        box (list): A list containing the height, width, left and top of the box.
    Returns:
        list: A list containing the x1, y1, x2 and y2 coordinates
    """
    return [box[2], box[3], box[2] + abs(box[1]), box[3] + abs(box[0])]


def calculate_iou(boxes_a, boxes_b):
    """
    Calculates the Intersection over Union (IoU) between two lists of boxes.
    Parameters:
        box_a (list): A list containing lists of coordinates on the form [height, width, left, top].
        box_b (list): A list containing lists of coordinates on the form [height, width, left, top].
    Returns:
        np.array: A matrix containing the IoU between the boxes in box_a and box_b.
    """

    boxes_a_copy = cp.deepcopy(boxes_a)
    boxes_b_copy = cp.deepcopy(boxes_b)

    for i in range(max(len(boxes_a_copy), len(boxes_b_copy))):
        if i < len(boxes_a_copy):
            boxes_a_copy[i] = format_box_to_iou(boxes_a_copy[i])
        if i < len(boxes_b_copy):
            boxes_b_copy[i] = format_box_to_iou(boxes_b_copy[i])
    return box_iou(torch.tensor(boxes_a_copy), torch.tensor(boxes_b_copy)).numpy()


def remove_overlapping_boxes(predicted_boxes, iou_threshold = 0.2):

    clean_predicted_boxes = []

    for page in predicted_boxes:

        if len(page) > 0:
            
            iou_matrix = calculate_iou(page, page)

            remove_indexes_page = []
            #Search only over the diagonl of the matrix

            num_rows, num_cols = iou_matrix.shape

            for i in range(num_rows):
                for j in range(i + 1, num_cols):
                    if iou_matrix[i][j] > iou_threshold:
                        
                        area_i = page[i][0] * page[i][1]
                        area_j = page[j][0] * page[j][1]

                        if area_i > area_j:
                            remove_indexes_page.append(i)
                        else:
                            remove_indexes_page.append(j)

            predicted_boxes_page = [box for i, box in enumerate(page) if i not in remove_indexes_page]
            clean_predicted_boxes.append(predicted_boxes_page)
        else:
            clean_predicted_boxes.append([])

    return clean_predicted_boxes


def ocr(image, run_tesseract, run_easyocr, languages, tess_config, elektronisk_tinglyst):
    """
    Perform OCR using Tesseract and/or EasyOCR based on the provided flags.
    
    Parameters:
    - image: The image to process.
    - run_tesseract (bool): Flag to run Tesseract OCR.
    - run_easyocr (bool): Flag to run EasyOCR.
    - languages (list): List of languages for EasyOCR.
    - tess_config (str): Configuration for Tesseract OCR.
    - elektronisk_tinglyst (bool): Additional configuration flag.
    
    Returns:
    - bounding_boxes (pd.DataFrame): DataFrame of bounding boxes from OCR.
    - text (str): Concatenated OCR text results.
    """
    
    text_parts = []
    bounding_boxes_parts = []
    
    if run_tesseract:
        text_tess, bounding_boxes_tess = apply_tesseractocr(image, tess_config, elektronisk_tinglyst)
        text_parts.append(text_tess)
        bounding_boxes_parts.append(bounding_boxes_tess)
    
    if run_easyocr:
        text_easy, bounding_boxes_easy = apply_easyocr(image, languages, elektronisk_tinglyst)
        text_parts.append(text_easy)
        bounding_boxes_parts.append(bounding_boxes_easy)

    text = " ".join(text_parts)
    bounding_boxes = pd.concat(bounding_boxes_parts, ignore_index=True) if bounding_boxes_parts else pd.DataFrame()

    return bounding_boxes, text

