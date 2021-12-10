import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prep_data(file_path : str,
              sets : list = None,
              train_perc : float = 0.7,
              y_label : str = "acc.overall.avg",
              y_cutoff : float = 100,
              one_hot : bool = False, # whether to apply one-hot encoding to categorical variables
              seed : int = 42,
             ):
    """
        Loads data in from the file_path, 
        removes entries with anomalous responses,
        applies custom encodings
        applies mean/mode imputation depending on type of variable.
    """
    df = load_data(file_path) # load raw data
    
    # select X, y below y_cutoff
    all_y = df[y_label]
    y = all_y[all_y<y_cutoff]
    if sets == None:
        all_X = df[set(df.columns)-{y_label}]
    else:
        all_X = select_sets(df, *sets)
    X = all_X[all_y<y_cutoff]
    
    # apply custom encodings
    X = custom_encodings(X)
    
    # split data into training and validation/testing (we will later divide validation and testing)
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_perc, random_state=seed)
    
    # divide up data into numeric and categoric variables do imputation
    num_cols = get_numeric_cols(X_train)
    cat_cols = get_object_cols(X_train)
    
    # !!! important not to impute means and modes from the test data!!!
    X_train, means_modes = train_mean_mode_impute(X_train, num_cols, cat_cols) # training imputation
    X_val_test = test_impute(X_val_test, means_modes) # test_val imputation using train means / modes
    
    # apply one-hot encoding if desired to categorical columns (have to do this post-imputation)
    if one_hot:
        X_train = pd.get_dummies(X_train, columns=cat_cols)
        X_val_test = pd.get_dummies(X_val_test, columns=cat_cols)
    
    # finally divide up X_val_test 
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5, random_state=seed)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, means_modes

def load_data(file_path : str):
    """ Utility function to load the data files and return pd dataframe """
    return pd.read_csv(file_path)

def select_sets(df : pd.DataFrame, 
                *sets : set, # as many as you want
               ):
    """ Selects the features in the union of the provided sets """
    union = set.union(*sets)
    return df[union]

def train_mean_mode_impute(df : pd.DataFrame,
                           num_cols : list,
                           cat_cols: list):
    """ 
        Imputes missing values as mean for numerical features and mode for categorical features.
        Importantly, this should only be used for the training data otherwise it is data leakage.
        This returns the mean/mode for each column which can then be used to fill in missing values
        in testing or validation data. 
    """
    means_modes = []
    for col in df.columns:
        mean_mode = -1
        if col in num_cols: # if numerical column
            mean_mode = df[col].mean()
            df[col].fillna(mean_mode, inplace=True)
        elif col in cat_cols: # if categorical column
            mean_mode = df[col].mode().iloc[0]
            df[col].fillna(mean_mode, inplace=True)
        else:
            print('Unclear column:', col)
        means_modes.append(mean_mode)
    return df, means_modes

def test_impute(df : pd.DataFrame,
                col_vals : list):
    """
       Fills missing values in col i with col_vals i. Ideally pass in the means_modes determined from the function above.
    """
    cols = df.columns
    for i in range(len(cols)):
        df[cols[i]].fillna(col_vals[i], inplace=True)
    return df

def to_numpy_cont(df : pd.DataFrame):
    """ 
       Converts a pandas Dataframe into a C contiguous numpy array
    """
    return np.ascontiguousarray(df.to_numpy())

def get_numeric_cols(df : pd.DataFrame):
    """ Returns the names of numeric columns """
    return list(df.select_dtypes([np.number]).columns)

def get_object_cols(df : pd.DataFrame):
    """ Returns the names of columns of type object (typically categorical) """
    return list(df.select_dtypes([object]).columns)
   

def custom_encodings(df : pd.DataFrame):
    pd.options.mode.chained_assignment = None
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
        elif col == "Tea intake":
            df = encode_tea_intake(df)
        elif col == "Cooked vegetable intake | Instance 0":
            df = encode_cooked_veg_intake(df)
        elif col == "Fresh fruit intake | Instance 0":
            df = encode_fresh_fruit_intake(df)
        elif col == "Oily fish intake | Instance 0":
            df = encode_oily_fish(df)
        elif col == "Salad / raw vegetable intake | Instance 0":
            df = encode_salad_intake(df)
        elif col == "Salt added to food":
            df = encode_added_salt(df)
        elif col == "Water intake | Instance 0":
            df = encode_added_salt(df)
        elif col == "Irritability | Instance 0":
            df = encode_irritability(df)
        elif col == "Miserableness | Instance 0":
            df = encode_mis(df)
        elif col == "Mood swings | Instance 0":
            df = encode_mood(df)
        elif col == "Sensitivity / hurt feelings | Instance 0":
            df =  encode_sensitivity(df)
        elif col == "Worrier / anxious feelings | Instance 0":
            df = encode_anxiety(df)
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
def encode_water_intake(df): 
    df["Water intake | Instance 0"].replace(
        {"Less than one" : 0,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Water intake | Instance 0"] = pd.to_numeric(df["Water intake | Instance 0"])
    return df

# Irritability | Instance 0 
def encode_irritability(df): 
    df["Irritability | Instance 0"].replace(
        {"No" : 0,
         "Yes": 1,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Irritability | Instance 0"] = pd.to_numeric(df["Irritability | Instance 0"])
    return df

# Miserableness | Instance 0
def encode_mis(df):
    df["Miserableness | Instance 0"].replace(
        {"No" : 0,
         "Yes": 1,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Miserableness | Instance 0"] = pd.to_numeric(df["Miserableness | Instance 0"])
    return df

# Mood swings | Instance 0
def encode_mood(df):
    df["Mood swings | Instance 0"].replace(
        {"No" : 0,
         "Yes": 1,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Mood swings | Instance 0"] = pd.to_numeric(df["Mood swings | Instance 0"])
    return df

# Sensitivity / hurt feelings | Instance 0
def encode_sensitivity(df):
    df["Sensitivity / hurt feelings | Instance 0"].replace(
        {"No" : 0,
         "Yes": 1,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Sensitivity / hurt feelings | Instance 0"] = pd.to_numeric(df["Sensitivity / hurt feelings | Instance 0"])
    return df

# Worrier / anxious feelings | Instance 0
def encode_anxiety(df):
    df["Worrier / anxious feelings | Instance 0"].replace(
        {"No" : 0,
         "Yes": 1,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Worrier / anxious feelings | Instance 0"] = pd.to_numeric(df["Worrier / anxious feelings | Instance 0"])
    return df

