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

def mean_mode_impute(df : pd.DataFrame,
                     num_cols : list,
                     cat_cols: list):
    """ Imputes missing values as mean for numerical features and mode for categorical features """
    for col in df.columns:
        if col in num_feats: # if numerical column
            df[col].fillna(df[col].mean(), inplace=True)
        elif col in cat_cols: # if categorical column
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
        else:
            print('Unclear column:', col)
    return df

def to_numpy_cont(X : pd.DataFrame):
    """ 
       Converts a pandas Dataframe into a C contiguous numpy array
    """
    return np.ascontiguousarray(X.to_numpy())

def custom_encodings(df : pd.DataFrame):
    for col in df:
        if col == "Frequency of stair climbing in last 4 weeks | Instance 0":
            df = encode_freq_stair(df)
        elif col == "Duration of walks | Instance 0":
            df = encode_duration_of_walks(df)
        elif col == "Time spent using computer | Instance 0":
            df = encode_time_computer(df)
        elif col == "Time spend outdoors in summer | Instance 0":
            df = encode_time_outdoors_summer(df)
        elif col == "Time spent outdoors in winter | Instance 0":
            df = encode_time_outdoors_winter(df)
        elif col == "Time spent watching television (TV) | Instance 0":
            df = encode_time_tele(df)
    
    return df

# CUSTOM ENCODINGS
# Frequency of stair climbing in last 4 weeks | Instance 0
def encode_freq_stair(df : pd.DataFrame):
    df["Frequency of stair climbing in last 4 weeks | Instance 0"].replace(
        {"None" : 0,
         "1-5 times a day" : 5, 
         "6-10 times a day" : 10,
         "11-15 times a day" : 15,
         "16-20 times a day" : 20,
         "More than 20 times a day" : 25,
         "Do not know" : np.nan,
         "Prefer not to answer" : np.nan},  
        inplace=True)
    df["Frequency of stair climbing in last 4 weeks | Instance 0"] = pd.to_numeric(df["Frequency of stair climbing in last 4 weeks | Instance 0"])
    return df

# Duration of walks | Instance 0
def encode_duration_of_walks(df : pd.DataFrame):
    df["Duration of walks | Instance 0"].replace(
        {"Do not know" : np.nan, 
         "Prefer not to answer" : np.nan},  
        inplace=True)
    df["Duration of walks | Instance 0"] = pd.to_numeric(df["Duration of walks | Instance 0"])
    return df

# Time spent using computer | Instance 0
def encode_time_computer(df : pd.DataFrame):
    df["Time spent using computer | Instance 0"].replace(
        {"Less than an hour a day" : 0.5, 
         "Prefer not to answer" : np.nan, 
         "Do not know" : np.nan},  
        inplace=True)
    df["Time spent using computer | Instance 0"] = pd.to_numeric(df["Time spent using computer | Instance 0"])
    return df

# Time spend outdoors in summer | Instance 0
def encode_time_outdoors_summer(df : pd.DataFrame):
    df["Time spend outdoors in summer | Instance 0"].replace(
        {"Less than an hour a day" : 0.5, 
         "Prefer not to answer" : np.nan, 
         "Do not know" : np.nan},  
        inplace=True)
    df["Time spend outdoors in summer | Instance 0"] = pd.to_numeric(df["Time spend outdoors in summer | Instance 0"])
    return df

# Time spent outdoors in winter | Instance 0
def encode_time_outdoors_winter(df : pd.DataFrame):
    df["Time spent outdoors in winter | Instance 0"].replace(
        {"Less than an hour a day" : 0.5, 
         "Prefer not to answer" : np.nan, 
         "Do not know" : np.nan},  
        inplace=True)
    df["Time spent outdoors in winter | Instance 0"] = pd.to_numeric(df["Time spent outdoors in winter | Instance 0"])
    return df

def encode_time_tele(df : pd.DataFrame):
    df["Time spent watching television (TV) | Instance 0"].replace(
        {"Less than an hour a day" : 0.5, 
         "Prefer not to answer" : np.nan, 
         "Do not know" : np.nan},  
        inplace=True)
    df["Time spent watching television (TV) | Instance 0"] = pd.to_numeric(df["Time spent watching television (TV) | Instance 0"])
    return df

