import pandas as pd
import regex
import re

def matches_within_one_edit(pattern, string):
    fuzzy_pattern = r'(?e)(?:' + pattern + '){e<=1}'
    match = regex.search(fuzzy_pattern, string, regex.IGNORECASE)
    return match is not None

def apply_hjemmel_search(bounding_boxes, debug_print=False):
    bbs_hjemmel = []

    table_header_map = {
        'avd.?de': {
            'left_side_adjustment': 2,
            'width_multiplier': 3.5,
        },
        'hjemmelshaver': {
            'left_side_adjustment': 2,
            'width_multiplier': 1.5,
        },
        'overdras': {
            'left_side_adjustment': 3.25,
            'width_multiplier': 3,
        },
        'til': {
            'left_side_adjustment': 3,
            'width_multiplier': 6.5,
        },
        'bortfestes': {
            'left_side_adjustment': 3.25,
            'width_multiplier': 3,
        }
    }
    MIN_HEADER_COLUMN_WIDTH = 320
    MAX_HEADER_COLUMN_WIDTH = 600
    MIN_HEADER_LEFT_SIDE_ADJUSTMENT = 40

    stopping_words = ['hjemmel', 'underskrift', 'avtaler', 's.?rskilt']
    second_row_must_contain_at_least_one = ['f.?dselsnr', 'sif(ft|tf)er']

    table_header_bbs = pd.DataFrame(columns=['height', 'width', 'left', 'top', 'text', 'type', 'table_header_key'])

    for index, row in bounding_boxes.iterrows():
        for key in table_header_map.keys():
            if matches_within_one_edit(key, row['text']):
                last_index = table_header_bbs.index[-1] if len(table_header_bbs) > 0 else -1
                table_header_bbs.loc[last_index + 1] = row
                table_header_bbs.loc[last_index + 1, 'table_header_key'] = key

    for index, row in table_header_bbs.iterrows():
        table_header_key = row['table_header_key']
        table_header_properties = table_header_map[table_header_key]
        left_side_adjustment = table_header_properties['left_side_adjustment']
        width_multiplier = table_header_properties['width_multiplier']

        adjusted_left = row['left'] - max(MIN_HEADER_LEFT_SIDE_ADJUSTMENT, (row['width'] / left_side_adjustment))
        adjusted_width = min(MAX_HEADER_COLUMN_WIDTH, max(MIN_HEADER_COLUMN_WIDTH, row['width'] * width_multiplier))
        top = row['top']
        height = row['height']

        header_bb = [height, adjusted_width, adjusted_left, top]
        # bbs_hjemmel.append(header_bb) # This adds the header as a bounding box for debugging purposes

        if debug_print:
            print(f"\nProcessing table header '{table_header_key}' at index {index}")


        tolerance_percentage_left = 0.1
        tolerance_percentage_top = 0.55
        tolerance_percentage_right = 0.1

        rows = []
        all_potential_bbs = []

        min_left_edge = adjusted_left - (adjusted_width * tolerance_percentage_left)
        max_right_edge = adjusted_left + adjusted_width + (adjusted_width * tolerance_percentage_right)

        has_double_header = False
        has_been_extended_to_navn = False

        # While first time or there are still rows to process
        while len(rows) == 0 or len(all_potential_bbs) > 0:

            if len(rows) == 0:
                previous_row = [header_bb]
                tolerance_percentage_bottom = 0.2
            else:
                previous_row = rows[-1]
                tolerance_percentage_bottom = 0.64

            if len(rows) == 2 and has_double_header:
                tolerance_percentage_bottom = 1.0

            MIN_ROW_HEIGHT = 40
            MAX_ROW_HEIGHT = 100
            previous_row_min_top = min([row[3] for row in previous_row])
            previous_row_max_bottom = max([previous_row_min_top + min(max(row[0], MIN_ROW_HEIGHT, row[0]*1.1), MAX_ROW_HEIGHT) for row in previous_row])
            previous_row_height = previous_row_max_bottom - previous_row_min_top

            min_top_edge = previous_row_max_bottom - (previous_row_height * tolerance_percentage_top)
            max_bottom_edge = previous_row_max_bottom + previous_row_height + (previous_row_height * tolerance_percentage_bottom)

            MIN_BB_WIDTH = 20
            MIN_BB_HEIGHT = 20
            MAX_BB_HEIGHT = 65
            MAX_DIFF_BETWEEN_NAVN_AND_RIGHT_EDGE = 150

            all_potential_bbs_extended = bounding_boxes[
                (bounding_boxes['left'] >= min_left_edge) &
                (bounding_boxes['top'] >= min_top_edge) &
                (bounding_boxes['width'] >= MIN_BB_WIDTH) &
                (bounding_boxes['height'] >= MIN_BB_HEIGHT) &
                (bounding_boxes['height'] <= MAX_BB_HEIGHT)
            ]

            all_potential_bbs = all_potential_bbs_extended.loc[
                (bounding_boxes['left'] + bounding_boxes['width'] <= max_right_edge) &
                (bounding_boxes['top'] + bounding_boxes['height'] <= max_bottom_edge)
            ]

            # This catches if the first row is empty
            if len(rows) == 1 and len(all_potential_bbs) == 0:
                all_potential_bbs = all_potential_bbs_extended.loc[
                    (bounding_boxes['left'] + bounding_boxes['width'] <= max_right_edge) &
                    (bounding_boxes['top'] + bounding_boxes['height'] <= max_bottom_edge + MIN_ROW_HEIGHT)]

            extended_bbs = all_potential_bbs_extended.loc[
                (bounding_boxes['top'] + bounding_boxes['height'] <= max_bottom_edge)
            ]
            extended_bbs_text_list = extended_bbs['text'].tolist()

            for index, row in extended_bbs.iterrows():
                # If the text is 'navn' and the bounding box is close to the right edge, it is likely a continuation of the previous column, so we should extend the right edge
                if not has_been_extended_to_navn and matches_within_one_edit('navn', row['text']) and abs(row['left'] + (row['width']/2) - max_right_edge) < MAX_DIFF_BETWEEN_NAVN_AND_RIGHT_EDGE:
                    max_right_edge = row['left'] + (row['width']/2)
                    has_been_extended_to_navn = True

            concatted_string_of_potential_bbs = all_potential_bbs['text'].str.cat(sep='').strip()

            # This prints the bounding box of the current row for debugging purposes
            if len(rows) == -1: # Change -1 to the index of the row you want to print
                bbs_hjemmel.append([max_bottom_edge - min_top_edge, max_right_edge - min_left_edge, min_left_edge, min_top_edge])

            if debug_print:
                print("Concatted string of potential bbs:", concatted_string_of_potential_bbs, "extended:", extended_bbs_text_list)

            concatted_string_of_potential_bbs = all_potential_bbs['text'].str.cat(sep='').strip()

            if len(rows) == 0:
                meets_subheader_criteria = False
                for must_contain_word in second_row_must_contain_at_least_one:
                    if matches_within_one_edit(must_contain_word, concatted_string_of_potential_bbs):
                        meets_subheader_criteria = True
                        break

                if not meets_subheader_criteria:
                    break

            else:
                if len(rows) == 1:
                    for must_contain_word in second_row_must_contain_at_least_one:
                        if matches_within_one_edit(must_contain_word, concatted_string_of_potential_bbs):
                            has_double_header = True
                            break

                # If there are multiple headers, we don't support that. Skip each row where regex matches
                if len(rows) > 1:
                    should_break = False
                    for must_contain_word in second_row_must_contain_at_least_one:
                        if matches_within_one_edit(must_contain_word, concatted_string_of_potential_bbs):
                            should_break = True
                            break
                    if should_break:
                        break

                MIN_CONCATTED_STRING_LENGTH = 5 if len(rows) == 1 or (len(rows) == 2 and has_double_header) else 7
                if len(concatted_string_of_potential_bbs) < MIN_CONCATTED_STRING_LENGTH:
                    break

                if not any(chr.isdigit() for chr in concatted_string_of_potential_bbs):
                    break

                # If it starts with a 9, it is an organization number unless the OCR has made a mistake
                if concatted_string_of_potential_bbs.startswith('9') and any(re.search(r'(A.?S|DA|BA|ASA|SE|ANS|kommune|stiftselse|SF)', s, re.IGNORECASE) for s in extended_bbs_text_list):
                    break

                should_break = False
                for stopping_word in stopping_words:
                    if matches_within_one_edit(stopping_word, concatted_string_of_potential_bbs):
                        should_break = True
                        break

                if should_break:
                    break

            rows.append(all_potential_bbs[['height', 'width', 'left', 'top']].values.tolist())

        # Remove first row, as it is the subheader (fødselsnummer etc.)
        if len(rows) > 0:
            if has_double_header:
                rows = rows[2:]
            else:
                rows = rows[1:]

        # Combine bbs within rows, using the leftmost left and the rightmost right.
        # Then take make bb using the last 45% of the bounding box
        row_bbs = []
        for row in rows:
            # row example: [[height, width, left, top], [height, width, left, top], ...]
            left = min([r[2] for r in row])
            top = min([r[3] for r in row])
            right = max([r[2] + r[1] for r in row])
            bottom = max([r[3] + r[0] for r in row])

            row_bbs.append([bottom - top, right - left, left, top])

        if len(row_bbs) == 0:
            continue

        min_left = min([row_bb[2] for row_bb in row_bbs])
        max_right = max([row_bb[2] + row_bb[1] for row_bb in row_bbs])
        width = (max_right - min_left)
        row_bbs_final = [[row_bb[0], width * 0.45, min_left + width * 0.55, row_bb[3]] for row_bb in row_bbs]
        bbs_hjemmel.extend(row_bbs_final)

    return bbs_hjemmel
