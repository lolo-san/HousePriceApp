import os
import csv
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

def create_preproc_pipe() -> Pipeline:
    # """
    # Create a pipeline for preprocessing data, with parrallel processing for 
    # numeric and categorical features
    # return: a pipeline object
    # """
    num_preproc_pipe = Pipeline([("Imputer",SimpleImputer())
                                ,("Scaling",StandardScaler())
                                ])

    cat_preproc_pipe = Pipeline([("imputer",SimpleImputer(strategy="most_frequent"))
                                ,("Encode",OneHotEncoder(drop="first", handle_unknown="ignore"))
                                ])
    preproc_pipe = ColumnTransformer([("NumPreproc",num_preproc_pipe,make_column_selector(dtype_include="number"))
                                    ,("CatPreproc",cat_preproc_pipe,make_column_selector(dtype_include="object"))
                                    ])

    return preproc_pipe
    
def create_model_pipe() -> Pipeline:
    # """
    # Create  training a model
    # return: a pipeline or model object
    # """
    model = VotingRegressor([("rand",RandomForestRegressor(min_samples_leaf=5))
                            ,("lin",LinearRegression())
                            ,("knn",KNeighborsRegressor())
                            ])
    return model

def save_pipe(model: Pipeline, filename: str) -> None:
    """
    Save the model (or Pipeline) to the models folder
    """
    if not os.path.exists("models"):
        os.makedirs("models")

    with open(os.path.join("models", filename),"wb") as file :
        joblib.dump(model, file)

def load_pipe(filename: str) -> Pipeline:
    """
    Load the model (or Pipeline) from the models folder
    """
    with open("../models/preproc.pickle","rb") as file :
        pipeline = joblib.load(file)

    return pipeline

def save_metrics(metrics: dict, filename: str) -> None:
    """
    Save the metrics to the metrics folder
    """
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    if os.path.exists(filename):
        with open(os.path.join("metrics", filename), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(metrics.values())
    else:
        with open(os.path.join("metrics", filename), "w") as file:
            writer = csv.writer(file)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())

def predict(data: pd.DataFrame, model: Pipeline) -> np.ndarray:
    """
    Make predictions using the trained model
    
    # WARNING : The data should be preprocessed before making predictions
    EXACTLY like the training data
    """
    pass
