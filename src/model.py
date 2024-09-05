from sklearn.pipeline import Pipeline

def create__preproc_pipe() -> Pipeline:
    """
    Create a pipeline for preprocessing data, with parrallel processing for 
    numeric and categorical features
    return: a pipeline object
    """
    pass
    
def create_model_pipe() -> Pipeline:
    """
    Create  training a model
    return: a pipeline or model object
    """
    pass


def save_model(model: Pipeline, filename: str) -> None:
    """
    Save the model to the models folder
    """
    pass


def save_metrics(metrics: dict, filename: str) -> None:
    """
    Save the metrics to the metrics folder
    """
    pass

def load_pipe(filename: str) -> Pipeline:
    """
    Load the model (or Pipeline) from the models folder
    """
    pass


def predict(data: pd.DataFrame, model: Pipeline) -> np.ndarray:
    """
    Make predictions using the trained model
    
    # WARNING : The data should be preprocessed before making predictions
    EXACTLY like the training data
    """
    pass
