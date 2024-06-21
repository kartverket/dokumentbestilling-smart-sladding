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



def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
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


def find_regex_matches(text):
    pattern_personummer = re.compile(r'(?:0[1-9]|[12][0-9]|3[01])(?:0[1-9]|1[0-2])\d{2}\s?\d{5}')
    pattern_dnummer = re.compile(r'4(?:[1-9]|[12][0-9]|3[01])(?:0[1-9]|1[0-2])\d{2}\s?\d{5}')

    patterns = [pattern_personummer, pattern_dnummer]
    categories = ['personnummer', 'dnummer']
    tagged_matches = []


    for pattern, tag in zip(patterns, categories):
        matches = re.findall(pattern, text)
        for index, match in enumerate(matches):
            tagged_matches.append([match, tag, index])

    return tagged_matches



def get_boxes_to_blur(tagged_matches, bounding_boxes):

    matches_list = []
    for i in tagged_matches:
        sep_matches = i[0].split(' ')
        for j in sep_matches:
            matches_list.append([j, i[1], i[2]])

    data = []
    for match in matches_list:
        for index, row in bounding_boxes.iterrows():
            pattern = re.compile(re.escape(match[0]), re.IGNORECASE)
            if pattern.search(row['text']):
                loc = (row['left'], row['top'], row['width'], row['height'])
        data.append([match[0], match[1], match[2], loc])
    return data
