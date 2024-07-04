import numpy as np
import scrapy
import pandas as pd
import PyPDF2
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os
import cv2


def get_pdf_dimensions(pdf_path):
    pdf = PyPDF2.PdfReader(open(pdf_path, "rb"))
    dimensions = []
    for page in range(len(pdf.pages)):
        media_box = pdf.pages[page].mediabox
        width = float(media_box.width)
        height = float(media_box.height)
        dimensions.append((width, height))
    return dimensions

def convert_pdf_to_images(pdf_path):
    dimensions = get_pdf_dimensions(pdf_path)
    images = []
    for page_number, (width, height) in enumerate(dimensions):
        # Convert page to image
        temp_images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1, size=(int(width), int(height)))
        images.extend(temp_images)  # Extend the list with the new images
    return images   

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def extract_text_and_bb_from_image(image):
    
    text = pytesseract.image_to_string(image, lang='nor')
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    bounding_boxes = data[['left', 'top', 'width', 'height', 'text']]
    bounding_boxes = bounding_boxes.dropna()
    
    #drop alle special characters
    bounding_boxes['text'] = bounding_boxes['text'].apply(remove_special_characters)

    return text, bounding_boxes

def check_controldigits(number):

    # Remove whitespace
    number = re.sub(r'\s*', '', number)

    # Split into digits
    digits = [int(d) for d in number]
    
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

def format_dnumber(dnumber):
    # Remove whitespace to get the original D-number
    dnumber = re.sub(r'\s*', '', dnumber)

    # Extract the day, month, year, and sequence parts
    day = int(dnumber[:2])
    month = int(dnumber[2:4])
    year = int(dnumber[4:6])
    sequence = dnumber[6:11]
    
    # The first digit of the day should be in the range 4-7
    if day < 40 or day > 71:
        return False
    
    # Validate control digits using the modulus 11 algorithm
    # Remove the added 4 to get the original day part for control digit calculation
    day -= 40
    
    # Concatenate the original parts
    personal_number = f'{day:02d}{month:02d}{year:02d}{sequence}'

    return personal_number

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
                match_f = format_dnumber(match)
                if check_controldigits(match_f):
                    tagged_matches.append([match, tag, index])
                    index += 1

            if tag == 'personnummer' and check_controldigits(match):
                tagged_matches.append([match, tag, index])
                index += 1

    return tagged_matches



def get_boxes_to_blur(tagged_matches, bounding_boxes):

    #[['020885 38717', 'personnummer', 0], ['190774 31058', 'personnummer', 1], ['020885 38717', 'personnummer', 2], ['190774 31058', 'personnummer', 3], ['190774 31058', 'personnummer', 4], ['020885 38717', 'personnummer', 5]]

    matches_list = []
    for i in tagged_matches:
        sep_matches = i[0].split(' ')
        if len(sep_matches) > 1:
            matches_list.append([sep_matches[1], i[1], i[2]])
        else:
            matches_list.append([sep_matches[0], i[1], i[2]])


    data = []
    bbs = []
    for match in matches_list:
        for index, row in bounding_boxes.iterrows():
            pattern = re.compile(re.escape(match[0]), re.IGNORECASE)
            if pattern.search(row['text']):
                loc = [row['height'], row['width'], row['top'], row['left']]
        data.append([match[0], match[1], match[2], loc])
        bbs.append(loc)
        
    return data, bbs

