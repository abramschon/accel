import numpy as np
import pandas as pd

def load_data(file_path : str):
    """ Utility function to load the data files and return pd dataframe """
    return pd.read_csv(file_path)

def select_sets(df : pd.DataFrame, 
                *sets : set, # as many as you want
               ):
    """ Selects the features in the union of the provided sets """
    union = set.union(*sets)
    return df[union]

def to_numpy_cont(X : pd.DataFrame):
    """ 
       Converts a pandas Dataframe into a C contiguous numpy array
    """
    return np.ascontiguousarray(X.to_numpy())
