import os
from file_utils import get_file


descriptors_collection_path='https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_1/anl_drug_molecular_descriptors/'
descriptors = 'combined_mordred_descriptors'
descriptors_url = descriptors_collection_path + "/" + descriptors

response_collection_path="https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_1/cancer_drug_response_prediction_dataset"
responses = "combined_single_response_agg"
responses_url = response_collection_path  + "/" + responses

rnaseq = "combined_rnaseq_data_lincs1000"
rnaseq_url = response_collection_path + "/" + rnaseq


data_dest = os.path.join("data", "raw" , "July2020")

get_file(descriptors + ".txt", descriptors_url, datadir=data_dest)

get_file(responses + ".txt", responses_url, datadir=data_dest)

get_file(rnaseq, rnaseq_url, datadir=os.path.join(data_dest, "lincs1000"))
