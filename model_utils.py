import re
import pytesseract
from pdf2image import convert_from_bytes
import requests
import fitz
import Levenshtein
import PIL
import numpy as np
import pandas as pd
import easyocr
from io import BytesIO
import torch
from torchvision.ops import box_iou
import copy as cp
import warnings


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
            ('født', 1), 
            ('fødselsdato', 3), 
            ('fodselsdato', 3), 
            ('fpnr', 1), 
            ('fødsnr', 1), 
            ('fodsnr', 1), 
            ('fødsnr', 1), 
            ('fodsnr', 1),
            ('identifikasjonsnummer', 3),
            ('fnrorgnr', 1),
            ('fødselsnrorganisasjonsnr', 4),
            ('fødselsnrorgnr', 3),
            ('fødselsorganisasjonsnummer', 4),
            ('identifikasjonsnummer', 3),
            ('fødselsnrforetaksnr', 3)]


def download_pdf(docid, base_url):
    """
    Download a PDF file from a URL.

    Parameters:
    docid (str): The document ID on form "aar_id_embete".
    base_url (str): The base URL of the API.

    Returns:
    bytes: The content of the PDF file.
    """

    url = f"{base_url}/{docid}.pdf"
    response = requests.get(url)

    if response.status_code == 200:
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)

    return response.content


def adjust_image_contrast(images, contrast_factor):
    """
    Adjust the contrast of a list of images.

    Parameters:
    images (list PIL.Image): The images to adjust.
    contrast_factor (float): The contrast factor.

    Returns:
    list PIL.Image: The adjusted images.
    """
    enhanced_images = []

    for image in images:
        # Create an ImageEnhance object
        enhancer = PIL.ImageEnhance.Contrast(image)

        # Enhance the image contrast
        enhanced_image = enhancer.enhance(contrast_factor)

        # Append the enhanced image to the list
        enhanced_images.append(enhanced_image)

    return enhanced_images


def convert_pdf_bytes_to_images(pdf_bytes, adjust_contrast = False, contrast_factor = 1.5):
    """
    Convert a PDF file to a list of images.

    Parameters:
    pdf_bytes (bytes): The PDF file as bytes.
    adjust_contrast (bool): Whether to adjust the contrast of the images.
    contrast_factor (float): The contrast factor.

    Returns:
    list: A list of images.
    tuple: A tuple containing the width and height of the images.
    """

    images = convert_from_bytes(pdf_bytes)

    if adjust_contrast:

        # Adjust the contrast of the images
        images = adjust_image_contrast(images, contrast_factor)

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


def apply_tesseractocr(image, languages = [], config = r'--oem 1 --psm 11', elektronisk_tinglyst = False):
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
        #drop alle special characters
        bounding_boxes['text'] = bounding_boxes['text'].apply(remove_special_characters)
        text = remove_special_characters(text)
    return text, bounding_boxes


def apply_easyocr(image, languages = ['no', 'en', 'da'], config = '', elektronisk_tinglyst = False):
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
    # Suppress FutureWarning related to torch.load
    warnings.filterwarnings("ignore", category=FutureWarning, 
                            message=r"You are using `torch.load` with `weights_only=False`")

    image = pil_to_cv2(image)
    reader = easyocr.Reader(languages)
    result = reader.readtext(image, x_ths = 0.01, y_ths = 0.01, width_ths = 0.01)
    result_df, text = format_easyocr_result_to_df(result)

    text = text.replace('\n', ' ')

    result_df = result_df.dropna()
    
    if not elektronisk_tinglyst:
        #drop alle special characters
        result_df['text'] = result_df['text'].apply(remove_special_characters)
        text = remove_special_characters(text)
    return text, result_df


def get_all_bbs(df):
    all_bbs = []
    for iter, row in df.iterrows():
        bb = [row['height'], row['width'], row['left'], row['top']]
        all_bbs.append(bb)

    return all_bbs


def check_controldigits(number):
    """
    Check if the control digits of a Norwegian personal number are correct.

    Parameters:
    number (str): The personal number to validate.

    Returns:
    bool: True if the control digits are correct, False otherwise.
    """

    # Remove whitespace
    number = re.sub(r'\s*', '', number)

    # Split into digits
    digits = [int(d) for d in number]

    if len(digits) == 11:
        # Weights for control number 1 (K1)
        weights_k1 = [3, 7, 6, 1, 8, 9, 4, 5, 2]

        # Calculate K1
        k1_sum = sum(d * w for d, w in zip(digits[:9], weights_k1))
        k1 = 11 - (k1_sum % 11)
        if k1 == 11:
            k1 = 0
        if k1 == 10:
            return False  

        # Weights for control number 2 (K2)
        weights_k2 = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

        # Calculate K2
        k2_sum = sum(d * w for d, w in zip(digits[:9] + [k1], weights_k2))
        k2 = 11 - (k2_sum % 11)
        if k2 == 11:
            k2 = 0
        if k2 == 10:
            return False 

        # Check if the control numbers are correct
        return digits[9] == k1 and digits[10] == k2
    
    return False


def format_dnumber(dnumber):
    """
    Format a D-number to a personal number.

    Parameters:
    dnumber (str): The D-number to format.

    Returns:
    bool: True if the D-number is a valid personal number, False otherwise.
    str: The formatted personal number.
    """

    is_personal_number = True
    # Remove whitespace to get the original D-number
    dnumber = re.sub(r'\s*', '', dnumber)

    # Extract the day, month, year, and sequence parts
    day = int(dnumber[:2])
    month = int(dnumber[2:4])
    year = int(dnumber[4:6])
    sequence = int(dnumber[6:11])
    
    # The first digit of the day should be in the range 4-7
    if day < 40 or day > 71:
        is_personal_number = False
    
    # Validate control digits using the modulus 11 algorithm
    # Remove the added 4 to get the original day part for control digit calculation
    day -= 40
    
    # Concatenate the original parts
    personal_number = f'{day:02d}{month:02d}{year:02d}{sequence}'

    return is_personal_number, personal_number


def find_matches(text):
    """
    Find matches in a text using regular expressions.

    Parameters:
    text (str): The text to search for matches.

    Returns:
    list: A list of matches with corresponding tag and index.
    """

    pattern_personummer = re.compile(r'(?:0[1-9]|[12][0-9]|3[01])\s*(?:0[1-9]|1[0-2])\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d') #obs: tillater ikke space mellom tallene i dag, måned, år
    pattern_dnummer = re.compile(r'[4,5,6,7]\s*(?:[0-9]|[12][0-9]|3[01])\s*(?:0[1-9]|1[0-2])\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d')

    patterns = [pattern_personummer, pattern_dnummer]
    categories = ['personnummer', 'dnummer']
    tagged_matches = []
    index = 0

    for pattern, tag in zip(patterns, categories):
        matches = re.findall(pattern, text)

        for match in matches:
            
            if tag == 'dnummer':
                is_personal_number, match_f = format_dnumber(match)
                if is_personal_number and check_controldigits(match_f):
                    tagged_matches.append([match, tag, index])
                    index += 1

            if tag == 'personnummer' and check_controldigits(match):
                tagged_matches.append([match, tag, index])
                index += 1

    return tagged_matches


def get_boxes_to_blur(tagged_matches, bounding_boxes):
    """
    Get bounding boxes for matches found in a text.

    Parameters:
    tagged_matches (list): A list of matches with corresponding tag and index.
    bounding_boxes (pd.DataFrame): A DataFrame containing the bounding boxes.

    Returns:
    list: A list of bounding boxes for the matches.
    """
    
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


def can_be_int(s):
    """
    Check if a string can be converted to an integer.

    Parameters:
    s (str): The string to check.

    Returns:
    str: 'whole_number' if the string can be a whole number, 'last_five' if the string is the last five digits of a number, False otherwise.
    """

    n_ints = 0
    s = s.replace(" ", "")
    s = re.findall(r'.', s)

    for char in s:
        try:
            integer = int(char)
            n_ints += 1
        except:
            continue
    
    if n_ints > 9 and n_ints < 14:
        return 'whole_number'
    if n_ints > 2 and len(s) == 5:
        return 'last_five'
    
    else:
        return False


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

    # Return 'last_five' if the string is 4 or 5 digits long and contains more than 2 digits
    if len(s) == 5:
        if int_count > 3:
            return 'last_five'
    
    # Return 'whole_number' if the string is 10 or 11 digits long and contains more than 9 digits
    if len(s) > 9 and len(s) < 14: 
        if int_count > 9:
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
                indexes.append(row[0])

    for index in indexes:

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


def remove_duplicated_boxes(predicted_boxes, iou_threshold = 0.2):

    clean_predicted_boxes = []

    for page in predicted_boxes:

        if len(page) > 0:
            
            iou_matrix = calculate_iou(page, page)
            print(iou_matrix)

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