import requests
import os
import pandas as pd

# 

def download_and_save_pdf(url, filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print("PDF downloaded successfully.")
    else:
        print("Failed to download file. HTTP Status Code:", response.status_code)


def download_all_documents(tinglyst_dokument_csv_path, save_folder_path):
    df_doc = pd.read_csv(tinglyst_dokument_csv_path)
    df_doc_ids =  df_doc["1980"].astype(str) + "_" + df_doc["14847"].astype(str) + "_" + df_doc["101"].astype(str)
    df_doc_ids[len(df_doc_ids)] = "1980_14847_101"

    try:
        os.mkdir(save_folder_path)
    except:
        pass
    for doc in df_doc_ids:
        url = f"https://dokumentbestilling-smart-sladding-manual.atkv3-dev.kartverket-intern.cloud/pantebok/{doc}.pdf"
        filename = save_folder_path + "/" + doc + ".pdf"
        download_and_save_pdf(url, filename)