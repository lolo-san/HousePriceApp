from .model import create__preproc_pipe,create_model_pipe
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging 


# TODO : Turn the print into logging
logging.basicConfig(level=logging.INFO)

loger = logging.getLogger(__name__)

def train():
    """
    
    """
    loger.info("Training the model")
    # Load the data
    data = load_data()
    preproc =  create__preproc_pipe()
    model = create_model_pipe()
    # Split the data into train and test sets
    # TODO : 
    train, test =
    
    # Fit the model
    train_preproc = preproc.fit_transform(train)
    model.fit(train)
    
    # Evaluate the model
    test_preproc = preproc.transform(test)
    model.predict(test)
    
    print("Mean Absolute Error: ", mean_absolute_error(test, model.predict(test)))
    print("Model trained successfully")
    
    # TODO : Save metrics  NOT AS LOGS
    # Save the MAE, MSE and R2 score in the metrics folder
    