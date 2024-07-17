import requests
import os
import pandas as pd
import evaluation_utils

def download_and_save_pdf(url, filename):
    """
    Download a PDF file from a URL and save it to the local file system.

    Parameters:
    url (str): The URL to download the PDF from.
    filename (str): The path to save the PDF file to.
    """

    # Download the PDF file
    response = requests.get(url)
    
    # Save the PDF file to the local file system
    # Check if the download was successful
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)

    return filename

def download_pdf(aar, id, embete, base_url):
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
    url = f"{base_url}/{doc}.pdf"
    response = requests.get(url)

    if response.status_code == 200:
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)

    return response.content


def download_all_documents(tinglyst_dokument_csv_path, save_folder_path, base_url):
    """
    Download all documents from a CSV file containing document information.

    Parameters:
    tinglyst_dokument_csv_path (str): The path to the CSV file containing document information.
    save_folder_path (str): The path to save the downloaded documents to.
    """

    # Load the CSV file containing document information
    df_doc = pd.read_csv(tinglyst_dokument_csv_path)

    # Combine the document information to create the document ID
    df_doc_ids =  df_doc["1980"].astype(str) + "_" + df_doc["14847"].astype(str) + "_" + df_doc["101"].astype(str)
    df_doc_ids[len(df_doc_ids)] = "1980_14847_101"

    # Create the save folder if it does not exist
    try:
        os.mkdir(save_folder_path)
        # Download and save each document
        for doc in df_doc_ids:
            url = f"{base_url}/{doc}.pdf"
            filename = save_folder_path + "/" + doc + ".pdf"
            download_and_save_pdf(url, filename)


    except FileExistsError:
        print("Folder already exists. Delete folder and try again.")
        pass


def organize_bounding_boxes(path_to_labels, path_to_bestilling_tinglyst_dokument, save_folder_path, save_organized_labels_path):
    """
    Organize bounding boxes from a CSV file containing labels and a CSV file containing document information.

    Parameters:
    path_to_labels (str): The path to the CSV file containing labels.
    path_to_bestilling_tinglyst_dokument (str): The path to the CSV file containing document information.

    Returns:
    pd.DataFrame: A DataFrame containing the organized bounding boxes.
    """
    
    # Load the CSV file containing the labels
    df_labels = pd.read_csv(path_to_labels)
    df_labels["dokument_nr_embete"] = df_labels["dokument_aar"].astype(str) + "_" + df_labels["dokument_nr"].astype(str) + "_" + df_labels["embete"].astype(str)

    #Load the CSV file containing all the document ids
    df_td = pd.read_csv(path_to_bestilling_tinglyst_dokument)
    df_td["dokument_nr_embete"] = df_td["1980"].astype(str) + "_" + df_td["14847"].astype(str) + "_" + df_td["101"].astype(str)

    #Create lists for all the bounding boxes and document ids
    bbs_all = []
    docs_all = []

    # Group the labels by document ID
    grouped_docs = df_labels.groupby('dokument_nr_embete')

    # Loop through the grouped documents
    for doc_id, doc_data in grouped_docs:
        bbs_doc = []

        try:
            #Get the page count for the document to add empty lists for pages without bounding boxes
            page_count = evaluation_utils.get_pdf_pagecount(save_folder_path + '/' + doc_id + '.pdf') 
            grouped_pages = doc_data.groupby('sidetall')

            # Loop through the grouped pages
            for page_id, page_data in grouped_pages:
                bbs = []
                for i, row in page_data.iterrows():

                    # Add the bounding box to the list
                    bbs.append([row["x"], row["y"], row["width"], row["height"]])

                # Add the bounding boxes for the page to the document
                bbs_doc.append(bbs)
            
            # Add empty lists for pages without bounding boxes
            for i in range(page_count - len(bbs_doc)):
                bbs_doc.append([])
            
            # Add the bounding boxes and document ID to the lists
            bbs_all.append(bbs_doc)
            docs_all.append(doc_id)

        except:
            print(f"Document {doc_id} not found.")
    
    # Make a new df from bbs_all and docs_all
    df = pd.DataFrame({
        "dokument_nr_embete" : docs_all,
        "bbs" : bbs_all
    })

    # Add missing rows to the df
    missing_rows = []
    for index, row in df_td.iterrows():
        if row["dokument_nr_embete"] not in df["dokument_nr_embete"].values:
            missing_rows.append({"dokument_nr_embete": row["dokument_nr_embete"], "bbs": []})
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    # Save the organized bounding boxes to a CSV file
    df.to_csv(save_organized_labels_path, index=False)

    return df



# This function requires that you have a folder inside the repo called valideringsset with the files bestilling_tinglyst_dokument.csv and labels.csv

def get_validation_set_main(tinglyst_dokument_cvs_path, path_to_labels, save_document_folder, base_url, save_organized_labels_path):
    """
    Get the validation set for the model.

    Returns:
    pd.DataFrame: A DataFrame containing the validation set.
    """

    # Download all documents
    download_all_documents(tinglyst_dokument_cvs_path, save_document_folder, base_url)

    # Organize the bounding boxes
    df = organize_bounding_boxes(path_to_labels, tinglyst_dokument_cvs_path,save_document_folder, save_organized_labels_path)

    return df

if __name__ == "__main__":
    df = get_validation_set_main("valideringssett/bestilling_tinglyst_dokument.csv", "valideringssett/labels.csv", "valideringssett/dokumenter" ,"https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok/", "valideringssett/organized_labels.csv")
