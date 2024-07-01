from pdf2image import convert_from_path
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
import requests

def download_and_save_pdf(url, filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)

    
#url = f""
#filename = f"dokumenter/{doc}.pdf"
#download_and_save_pdf(url, filename)


organized_labels_path = pd.read_csv("../valideringssett/organized_data.csv")

def get_pdf_dimensions(pdf_path):
    pdf = PyPDF2.PdfReader(open(pdf_path, "rb"))
    dimensions = []
    for page in range(len(pdf.pages)):
        media_box = pdf.pages[page].mediabox
        width = float(media_box.width)
        height = float(media_box.height)
        dimensions.append((width, height))
    return dimensions

def get_images_and_bb_from_index(labels_df, idx):
    row = labels_df.iloc[idx]
    doc = row["dokument_nr_embete"]
    bbs = row["bounding_boxes"]
    bbs = eval(bbs)
    filename = f"valideringssett/dokumenter/{doc}.pdf"

    dimensions = get_pdf_dimensions(filename)
    images = []
    for page_number, (width, height) in enumerate(dimensions):
        # Convert page to image
        temp_images = convert_from_path(filename, first_page=page_number+1, last_page=page_number+1, size=(int(width), int(height)))
        images.extend(temp_images)  # Extend the list with the new images
    return images, bbs


def visualize_bounding_boxes(df, idx):
    images, bbs = get_images_and_bb_from_index(df, idx)
    if bbs == []:
        print("No bounding boxes for this document")
        return
    for i, image in enumerate(images):
        if i >= len(bbs):
            continue
        fig, ax = plt.subplots(figsize=(20, 20))
        #Display image
        ax.imshow(image)
        #Iterate over all bounding boxes
        for bb in bbs[i]:
            #Create a rectangle patch
            rect = plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], edgecolor='r', facecolor="none")
            #Add the rectangle to the axes
            ax.add_patch(rect)
