import re
import pytesseract
from pdf2image import convert_from_bytes
import requests


def download_pdf(aar, id, embete):
    """
    Download a PDF file from a URL.

    Parameters:
    aar (int): The year of the document.
    id (int): The ID of the document.
    embete (int): The office of the document.

    Returns:
    bytes: The content of the PDF file.
    """

    doc = f"{aar}_{id}_{embete}"
    url = f"https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok/{doc}.pdf"
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

def remove_special_characters(text):
    """
    Remove special characters from a text.

    Parameters:
    text (str): The text to process.

    Returns:
    str: The text without special characters.
    """
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def extract_text_and_bb_from_image(image, config = r'--oem 3 --psm 11'):
    """
    Extract text and bounding boxes from an image.

    Parameters:
    image (PIL.Image?): The image to process.

    Returns:
    str: The extracted text.
    pd.DataFrame: A DataFrame containing the bounding boxes.
    """

    text = pytesseract.image_to_string(image, lang='nor', config=config)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, lang='nor', config=config)
    bounding_boxes = data[['left', 'top', 'width', 'height', 'text']]
    bounding_boxes = bounding_boxes.dropna()
    
    #drop alle special characters
    bounding_boxes['text'] = bounding_boxes['text'].apply(remove_special_characters)
    text = remove_special_characters(text)
    return text, bounding_boxes

def check_controldigits(number):

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
    pattern_personummer = re.compile(r'(?:0[1-9]|[12][0-9]|3[01])\s*(?:0[1-9]|1[0-2])\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d') #obs: tillater ikke space mellom tallene i dag, måned, år
    pattern_dnummer = re.compile(r'[4,5,6,7]\s*(?:[1-9]|[12][0-9]|3[01])\s*(?:0[1-9]|1[0-2])\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d')

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
    matches_list = []
    for i in tagged_matches:
        sep_matches = i[0].split(' ')
        matches_list.append([sep_matches[-1], i[1], i[2]])

    bbs = []
    for match in matches_list:
        pattern = re.compile(re.escape(match[0]), re.IGNORECASE)
        # Initialize a list to collect bounding boxes for each match
        match_bbs = []

        for index, row in bounding_boxes.iterrows():
            if pattern.search(row['text']):
                loc = [row['height'], row['width'], row['left'], row['top']]
                match_bbs.append(loc)
        
        # Extend the main bounding boxes list with all matches found for this pattern
        bbs.extend(match_bbs)

    bbs_clean = []
    for bb in bbs:
        if bb not in bbs_clean:
            bbs_clean.append(bb)

    return bbs_clean
