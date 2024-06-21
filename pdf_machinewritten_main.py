import smart_sladding_ml.pdf_machinewritten_utils as pmu

pdf_path = ''

def main(pdf_path):

    images = pmu.convert_pdf_to_images(pdf_path)

    for i, image in enumerate(images):
        text, bounding_boxes = pmu.extract_text_and_bb_from_image(image)
        tagged_matches = pmu.find_regex_matches(text)
        data = pmu.get_boxes_to_blur(tagged_matches, bounding_boxes)
        print(data)

main(pdf_path)
