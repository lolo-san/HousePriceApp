from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging 
from datetime import datetime

from data import load_data, clean_data
from model import create_preproc_pipe, create_model_pipe, save_metrics, save_pipe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("House Prices Prediction")

def train():
    """
    
    """
    # Load the data
    logger.info("Loading the data")
    data = load_data()

    # Clean the data
    logger.info("Cleaning the data")
    df = clean_data(data)

    # Create the preprocessing pipeline
    logger.info("Creating the preprocessing pipeline")
    preproc = create_preproc_pipe()

    # Create the model pipeline
    logger.info("Creating the model pipeline")
    model = create_model_pipe()

    # Split the data into train and test sets
    train, test = train_test_split(df,random_state=42)

    y_train = train.pop("SalePrice")
    y_test = test.pop("SalePrice")

    # Fit the model
    logger.info("Fitting the model")
    train_preproc = preproc.fit_transform(train)
    model.fit(train_preproc, y_train)
    
    # Evaluate the model
    logger.info("Evaluating the model")
    test_preproc = preproc.transform(test)
    result = model.predict(test_preproc)
    mae = mean_absolute_error(y_test, result)
    mse = mean_squared_error(y_test, result)
    r2 = r2_score(y_test, result)
    logger.info("Mean Absolute Error: %2f", mae)
    logger.info("Mean Squared Error: %2f", mse)
    logger.info("R Square Score: %2f", r2)
    logger.info("Model trained successfully")
    
    time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    metrics = {"Time":time, "MAE":mae, "MSE":mse, "R2":r2}
    
    # Save the model's metrics to a csv file
    save_metrics(metrics, "metrics.csv")
    logger.info("Metrics saved successfully")

    # Save the model to a file
    logger.info("Saving the model")
    save_pipe(preproc, "preproc.pickle")
    save_pipe(model, "model.pickle")
    logger.info("Done")

if __name__ == "__main__":
    train()