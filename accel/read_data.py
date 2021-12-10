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

# Tea intake
def encode_tea_intake(df: pd.DataFrame):
    df["Tea intake"].replace(
        {"Less than one" : 0,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Tea intake"] = pd.to_numeric(df["Tea intake"])
    return df

#Cooked vegetable intake | Instance 0
def encode_cooked_veg_intake(df):
    df["Cooked vegetable intake | Instance 0"].replace(
        {"Less than one" : 0,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Cooked vegetable intake | Instance 0"] = pd.to_numeric(df["Cooked vegetable intake | Instance 0"])
    return df

#Fresh fruit intake | Instance 0
def encode_fresh_fruit_intake(df):
    df["Fresh fruit intake | Instance 0"].replace(
        {"Less than one" : 0,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Fresh fruit intake | Instance 0"] = pd.to_numeric(df["Fresh fruit intake | Instance 0"])
    return df

# Oily fish intake | Instance 0
def encode_oily_fish(df):
    df["Oily fish intake | Instance 0"].replace(
        {"Never" : 0,
         "Less than once a week": 0.5,
         "Once a week": 1,
         "2-4 times a week": 3,
         "5-6 times a week": 5.5,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Oily fish intake | Instance 0"] = pd.to_numeric(df["Oily fish intake | Instance 0"])
    return df

#Salad / raw vegetable intake | Instance 0
def encode_salad_intake(df):
    df["Salad / raw vegetable intake | Instance 0"].replace(
        {"Less than one" : 0,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Salad / raw vegetable intake | Instance 0"] = pd.to_numeric(df["Salad / raw vegetable intake | Instance 0"])
    return df

# Salt added to food
def encode_added_salt(df):
    df["Salt added to food"].replace(
        {"Never/rarely" : 0,
         "Sometimes": 1,
         "Usually": 2,
         "Always": 3,
         "Prefer not to answer": np.nan,
        }, inplace=True
    )
    df["Salt added to food"] = pd.to_numeric(df["Salt added to food"])
    return df

# Water intake | Instance 0
def water_intake(df): 
    df["Water intake | Instance 0"].replace(
        {"Less than one" : 0,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Water intake | Instance 0"] = pd.to_numeric(df["Water intake | Instance 0"])
    return df

