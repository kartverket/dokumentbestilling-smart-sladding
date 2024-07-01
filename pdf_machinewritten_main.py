import pdf_machinewritten_utils as pmu

pdf_path = 'smart_sladding_ml/valideringssett/dokumenter/2023_73325_200.pdf'

def not_main(pdf_path):

    images = pmu.convert_pdf_to_images(pdf_path)

    predicted_boxes = []

    for i, image in enumerate(images):
        text, bounding_boxes = pmu.extract_text_and_bb_from_image(image)
        tagged_matches = pmu.find_matches(text)
        data, bbs = pmu.get_boxes_to_blur(tagged_matches, bounding_boxes)
        predicted_boxes.append(bbs)

    return images, predicted_boxes

images, predicted_boxes = not_main(pdf_path)
