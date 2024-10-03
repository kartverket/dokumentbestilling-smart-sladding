import cv2
import easyocr
import matplotlib.pyplot as plt

from pdf2image import convert_from_path

'''
Utils file to evaluate the OCR model.
Generates text next to bounding boxes on the image.

Currently only in use for debugging purposes.
'''

def draw_bounding_boxes(image, detections, threshold=0.25):

    for bbox, text, score in detections:

        if score > threshold:

            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)

            cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 0, 0), 2)

image_path = "valideringssett/all_documents/2013_98_200.pdf"
images = convert_from_path(image_path, fmt='jpeg')
reader = easyocr.Reader(['no', 'en', 'da'], gpu=False)


def contains_at_least_x_numbers(s, x):
    count = sum(c.isdigit() for c in s)
    return count >= x

# Loop over each image (page) and perform OCR
for page_num, image in enumerate(images, start=1):
    # Perform OCR on the image
    result = reader.readtext(image)

    # Display the image (optional)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # Draw bounding boxes and text on the image
    for bbox, text, confidence in result:
        # Extract the top-left and bottom-right coordinates of the bounding box
        top_left = tuple(bbox[0])
        bottom_right = tuple(bbox[2])

        if not contains_at_least_x_numbers(text, 5):
            continue

        # Draw the rectangle
        plt.gca().add_patch(plt.Rectangle(top_left, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1],
                                          edgecolor='red', facecolor='none', linewidth=1))

        # Annotate the text
        t = plt.text(top_left[0], top_left[1] - 10, text, fontsize=5, color='white', weight='bold')
        t.set_bbox(dict(facecolor='red', alpha=1, edgecolor='red', pad=0))

    # Show the image with annotations
    plt.title(f'Page {page_num}')
    plt.show()

    # Print the recognized text
    print(f'--- Text from Page {page_num} ---')
    for bbox, text, confidence in result:
        print(f'Text: {text} | Confidence: {confidence:.2f}')
    print('\n')
