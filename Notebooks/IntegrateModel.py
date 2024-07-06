import pandas as pd
import pickle 
import sklearn 
import requests
import json
from pytrials.client import ClinicalTrials
import re
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
import time
import ast

# code to integrate Regression model into the website
model_file_path = './project_new/trial_something/regression_model.pkl'
preprocessor_file_path =  './project_new/trial_something/preprocessor.pkl'

# Load processor
with open(preprocessor_file_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Load model
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

def divide_by_100(x):
    return x/100

def get_drug_disease_data(drug, disease):
    ctg_studies_path = './project_new/trial_something/ctg_studies.csv'
    ctg_studies = pd.read_csv(ctg_studies_path)
    ctg_studies["Conditions"] = ctg_studies["Conditions"].str.upper()
    ctg_studies["Interventions"] = ctg_studies["Interventions"].str.upper()
    ctg_studies["Conditions"] = ctg_studies["Conditions"].fillna('').astype(str)
    ctg_studies["Interventions"] = ctg_studies["Interventions"].fillna('').astype(str)

sample_data = pd.DataFrame({
    'Indication': ['STROKE'],
    'Num Studies': [10],
    'Total females in studies': [100],
    'Total males in studies': [200],
    'Number of participants in most relevant studies': [30],
    'Number of female participants in most relevant studies': [10],
    'Number of male participants in most relevant studies': [20]
})

sample_data['Female proportion in studies'] = sample_data['Total females in studies']/ sample_data['Total males in studies']+ sample_data['Total females in studies']
sample_data['Male proportion in studies'] = sample_data['Total males in studies']/ sample_data['Total males in studies']+ sample_data['Total females in studies']
sample_data['Proportion of females in most relevant studies'] = sample_data['Number of female participants in most relevant studies']/ sample_data['Number of participants in most relevant studies']
sample_data['Proportion of males in most relevant studies'] = sample_data['Number of male participants in most relevant studies']/ sample_data['Number of participants in most relevant studies']
transformed_data = preprocessor.transform(sample_data)

transformed_columns = (
    preprocessor.transformers_[0][1].get_feature_names_out(['Indication']).tolist() +
    [
        'Total females in studies', 
        'Total males in studies',
        'Female proportion in studies', 
        'Male proportion in studies',
        'Proportion of females in most relevant studies', 
        'Proportion of males in most relevant studies',
        'Num Studies',
        'Number of participants in most relevant studies', 
        'Number of female participants in most relevant studies',
        'Number of male participants in most relevant studies'
    ]
)
transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
transformed_df.drop(columns=['Female proportion in studies','Male proportion in studies', 'Proportion of females in most relevant studies', 'Proportion of males in most relevant studies' ],  inplace=True)
prediction = model.predict(transformed_df)*100
print(prediction)
