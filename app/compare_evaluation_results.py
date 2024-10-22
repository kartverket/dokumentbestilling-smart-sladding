import pandas as pd
import logging

def compare_results(path1, path2, version1='1.11', version2='1.12', filter_docids_file_ref=None):
    # Load results from both models
    results1 = pd.read_csv(f'{path1}/results.csv')
    results2 = pd.read_csv(f'{path2}/results.csv')

    # If a filter file is provided, read the document IDs to include
    if filter_docids_file_ref:
        with open(filter_docids_file_ref, 'r') as f:
            doc_ids = set(line.strip() for line in f)
        # Filter results to include only the specified documents
        results1 = results1[results1['Document'].astype(str).isin(doc_ids)]
        results2 = results2[results2['Document'].astype(str).isin(doc_ids)]

    # Group by 'Document' and sum up the TP, FP, FN, and Time(s) for each model
    agg_funcs = {'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'Time(s)': 'sum'}
    grouped1 = results1.groupby('Document').agg(agg_funcs).reset_index()
    grouped2 = results2.groupby('Document').agg(agg_funcs).reset_index()

    # Merge the two results on 'Document' with version-specific suffixes
    merged_results = pd.merge(
        grouped1, grouped2, on='Document',
        suffixes=(f'_{version1}', f'_{version2}')
    )

    # Decide which model is better for each document based on TP, FP, FN
    def compare_models(row):
        tp1 = row[f'TP_{version1}']
        tp2 = row[f'TP_{version2}']
        fp1 = row[f'FP_{version1}']
        fp2 = row[f'FP_{version2}']
        fn1 = row[f'FN_{version1}']
        fn2 = row[f'FN_{version2}']

        # Higher TP and lower FP and FN are considered better
        if tp1 > tp2:
            return version1
        elif tp1 < tp2:
            return version2
        else:
            # If TP are equal, check FP
            if fp1 < fp2:
                return version1
            elif fp1 > fp2:
                return version2
            else:
                # If FP are also equal, check FN
                if fn1 < fn2:
                    return version1
                elif fn1 > fn2:
                    return version2
                else:
                    return 'Equal'

    merged_results['Better_Model'] = merged_results.apply(compare_models, axis=1)

    # Calculate processing time differences
    merged_results['Time_Difference(s)'] = (
            merged_results[f'Time(s)_{version2}'] - merged_results[f'Time(s)_{version1}']
    )

    # Rearrange columns to group metrics under each model
    summary_table = merged_results[[
        'Document',
        f'TP_{version1}', f'FP_{version1}', f'FN_{version1}', f'Time(s)_{version1}',
        f'TP_{version2}', f'FP_{version2}', f'FN_{version2}', f'Time(s)_{version2}',
        'Better_Model', 'Time_Difference(s)'
    ]]

    # Create MultiIndex columns for better grouping
    summary_table.columns = pd.MultiIndex.from_tuples([
        ('', 'Document'),
        (version1, 'TP'), (version1, 'FP'), (version1, 'FN'), (version1, 'Time(s)'),
        (version2, 'TP'), (version2, 'FP'), (version2, 'FN'), (version2, 'Time(s)'),
        ('', 'Better_Model'), ('', 'Time_Difference(s)')
    ])

    # Set 'Document' as the index for better readability
    summary_table.set_index(('', 'Document'), inplace=True)

    # Print the summary table
    logging.info("Summary of Model Comparisons:")
    logging.info(summary_table)

    # Use MultiIndex column names for calculations and selections
    # Calculations for overall insights
    total_docs = len(summary_table)
    better_v1 = (summary_table[('', 'Better_Model')] == version1).sum()
    better_v2 = (summary_table[('', 'Better_Model')] == version2).sum()
    equal = (summary_table[('', 'Better_Model')] == 'Equal').sum()

    logging.info(f"\nTotal Documents Compared: {total_docs}")
    logging.info(f"Documents where Model {version1} is better: {better_v1}")
    logging.info(f"Documents where Model {version2} is better: {better_v2}")
    logging.info(f"Documents where both models perform equally: {equal}")

    avg_time_v1 = summary_table[(version1, 'Time(s)')].mean()
    avg_time_v2 = summary_table[(version2, 'Time(s)')].mean()
    logging.info(f"\nAverage Processing Time for Model {version1}: {avg_time_v1:.2f}s")
    logging.info(f"Average Processing Time for Model {version2}: {avg_time_v2:.2f}s")

    # Identify documents with significant discrepancies
    discrepancies = summary_table[summary_table[('', 'Better_Model')] != 'Equal']

    # Split discrepancies into two tables based on which model is better
    better_v1_docs = discrepancies[discrepancies[('', 'Better_Model')] == version1]
    better_v2_docs = discrepancies[discrepancies[('', 'Better_Model')] == version2]

    logging.info(f"\nDocuments where Model {version1} is better:")
    logging.info(better_v1_docs[[
        (version1, 'TP'), (version1, 'FP'), (version1, 'FN'),
        (version2, 'TP'), (version2, 'FP'), (version2, 'FN'),
    ]])

    logging.info(f"\nDocuments where Model {version2} is better:")
    logging.info(better_v2_docs[[
        (version1, 'TP'), (version1, 'FP'), (version1, 'FN'),
        (version2, 'TP'), (version2, 'FP'), (version2, 'FN'),
    ]])

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    compare_results(
        "valideringssett/results/1/1.16",
        "valideringssett/results/1/1.17",
        version1='1.16',
        version2='1.17',
        filter_docids_file_ref='valideringssett/filter_lists/only_docs_with_hjemmel.txt'
    )
