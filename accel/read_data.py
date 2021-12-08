import pandas as pd
from sklearn.model_selection import train_test_split

def load_xy(file_path : str, 
            x_names : [str, ...],
            y_name : str
           ):
    """ Loads data and selects provided feature and response names """
    data = pd.read_csv(file_path)
    X = data[x_names]
    y = data[y_name]
    return X, y

def load_data(file_path : str):
    """ Utility function to load the data files and return pd dataframe """
    return pd.read_csv(file_path)

def dataset_split(X : pd.DataFrame, 
                  y : pd.Series, 
                  perc_test : float):
    """ 
        Function to split the dataset into train and test 
        where perc_test is how the percentage of the total data used as test data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perc_test)
    return X_train, X_test, y_train, y_test

