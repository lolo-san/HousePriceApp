import os
import pandas as pd
import urllib.request

DATA_SET_URL = "https://storage.googleapis.com/schoolofdata-datasets/Data-Engineering.Production-Machine-Learning-Code/train.csv"
DATA_SET_FILENAME = "data/train.csv"

def load_data()-> pd.DataFrame:
    # """
    # Try to load the data from the data folder (localy) 
    # If data doesn't exist, fetch it from the web and save it in the data folder
    # In the end, return the data as a pandas DataFrame
    # """
    if not os.path.exists(DATA_SET_FILENAME):
        urllib.request.urlretrieve(DATA_SET_URL, DATA_SET_FILENAME)

    with open(DATA_SET_FILENAME, 'r') as file:
        data = pd.read_csv(file)

    return data

def clean_data(data: pd.DataFrame)-> pd.DataFrame:
    # """
    # Clean the data by removing missing values and duplicates
    # """
    return data.dropna(axis=1)
