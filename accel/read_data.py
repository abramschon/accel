import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prep_data(file_path : str,
              sets : list = None,
              train_perc : float = 0.7,
              y_label : str = "acc.overall.avg",
              y_cutoff : float = 100,
              normalise : bool = False, # whether to transform numerical variables to z-vals
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
    elif len(sets) == 1:
        all_X = df[sets[0]]
    else: # multiple sets
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
    
    # standardise numerical variables
    if normalise:
        scaler = StandardScaler().fit(X_train[num_cols]) # only train based on training data
        X_train[num_cols] = scaler.transform(X_train[num_cols])
        X_val_test[num_cols] = scaler.transform(X_val_test[num_cols])
    
    # apply one-hot encoding if desired to categorical columns (have to do this post-imputation)
    if one_hot:
        X_train = pd.get_dummies(X_train, columns=cat_cols)
        X_val_test = pd.get_dummies(X_val_test, columns=cat_cols)
        # ensure they have the same columns ! (think of better strategy for handling this)
        # this is because there could be categorical columns which disappear when we get dummies
        common_cols = set.intersection(set(X_train.columns), set(X_val_test.columns))
        X_train = X_train[common_cols]
        X_val_test = X_val_test[common_cols]
    
    # finally divide up X_val_test 
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5, random_state=seed)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, means_modes

def load_data(file_path : str):
    """ Utility function to load the data files and return pd dataframe """
    return pd.read_csv(file_path)

def select_sets(df : pd.DataFrame, 
                *sets : set, # must be more than one set
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
#         elif col == "Oily fish intake | Instance 0":
#             df = encode_oily_fish(df)
        elif col == "Salad / raw vegetable intake | Instance 0":
            df = encode_salad_intake(df)
        elif col == "Salt added to food":
            df = encode_added_salt(df)
        elif col == "Water intake | Instance 0":
            df = encode_water_intake(df)
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
        elif col == "Average total household income before tax | Instance 0":
            df = encode_household_income(df)
        elif col == "Number of vehicles in household | Instance 0":
            df = encode_num_vehicles(df)
        elif col == "Frequency of stair climbing in last 4 weeks | Instance 0":
            df = encode_stair_climbing(df)
        elif col == "Frequency of tiredness / lethargy in last 2 weeks | Instance 0":
            df = encode_tiredness(df)
        elif col == "IPAQ activity group | Instance 0":
            df = encode_ipaq(df)
        elif col == "Alcohol drinker status | Instance 0":
            df = encode_alc_drinker(df)
        elif col == "Alcohol intake frequency. | Instance 0":
            df = encode_alc_freq(df)
        elif col == "Length of mobile phone use | Instance 0":
            df = encode_mob_length(df)
        elif col == "Sleep duration | Instance 0":
            df = encode_sleep_dur(df)
        elif col == "Smoking status | Instance 0":
            df = encode_smoke_stat(df)
        elif col == "Weekly usage of mobile phone in last 3 months | Instance 0":
            df = encode_mob_use(df)
        elif col == "Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0":
            df = encode_blood_clot(df)
        elif col == "Cancer diagnosed by doctor | Instance 0":
            df = encode_cancer(df)
        elif col == "Chest pain or discomfort | Instance 0":
            df = encode_chest_pain(df)
        elif col == "Diabetes diagnosed by doctor | Instance 0":
            df = encode_diabetes(df)
        elif col == "Fractured/broken bones in last 5 years | Instance 0":
            df = encode_bones(df)
        elif col == "Mouth/teeth dental problems | Instance 0":
            df = encode_dental(df)
        elif col == "Other serious medical condition/disability diagnosed by doctor | Instance 0":
            df = encode_other_condition(df)
        elif col == "Overall health rating | Instance 0":
            df = encode_health_rating(df)
        elif col == "Vascular/heart problems diagnosed by doctor | Instance 0":
            df = encode_heart(df)
        elif col == "Age started wearing glasses or contact lenses":
            df = encode_glasses_age(df)
        elif col == "Breastfed as a baby | Instance 0":
            df = encode_breastfed(df)
        elif col == "Getting up in morning | Instance 0":
            df = encode_getting_up(df)
        elif col == "How are people in household related to participant | Instance 0":
            df = encode_household_relations(df)
        elif col == "Number in household | Instance 0":
            df = encode_num_in_household(df)
        elif col == "Wears glasses or contact lenses | Instance 0":
            df = encode_glasses(df)

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

# Time spent watching television (TV) | Instance 0
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
         "Once or more daily": 7,
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

# Salt added to food | Instance 0
def encode_added_salt(df):
    df["Salt added to food | Instance 0"].replace(
        {"Never/rarely" : 0,
         "Sometimes": 1,
         "Usually": 2,
         "Always": 3,
         "Prefer not to answer": np.nan,
        }, inplace=True
    )
    df["Salt added to food | Instance 0"] = pd.to_numeric(df["Salt added to food | Instance 0"])
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

# Average total household income before tax | Instance 0
def encode_household_income(df):
    df["Average total household income before tax | Instance 0"].replace(
        {"Less than 18,000" : 1,
         "18,000 to 30,999": 2,
         "31,000 to 51,999" : 3,
         "52,000 to 100,000": 4,
         "Greater than 100,000": 5,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Average total household income before tax | Instance 0"] = pd.to_numeric(df["Average total household income before tax | Instance 0"])
    return df

# Number of vehicles in household | Instance 0
def encode_num_vehicles(df):
    df["Number of vehicles in household | Instance 0"].replace(
        {"None": 0,
         "One" : 1,
         "Two": 2,
         "Three" : 3,
         "Four or more": 4,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Number of vehicles in household | Instance 0"] = pd.to_numeric(df["Number of vehicles in household | Instance 0"])
    return df

# Frequency of stair climbing in last 4 weeks | Instance 0
def encode_stair_climbing(df):
    df["Frequency of stair climbing in last 4 weeks | Instance 0"].replace(
        {"None": 0,
         "1-5 times a day" : 5,
         "6-10 times a day": 10,
         "11-15 times a day" : 15,
         "16-20 times a day" : 20,
         "More than 20 times a day": 30,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Frequency of stair climbing in last 4 weeks | Instance 0"] = pd.to_numeric(df["Frequency of stair climbing in last 4 weeks | Instance 0"])
    return df

# Frequency of tiredness / lethargy in last 2 weeks | Instance 0
def encode_tiredness(df):
    df["Frequency of tiredness / lethargy in last 2 weeks | Instance 0"].replace(
        {"Not at all": 0,
         "Several days" : 4,
         "More than half the days": 7,
         "Nearly every day" : 12,
         "Prefer not to answer": np.nan,
         "Do not know": np.nan
        }, inplace=True
    )
    df["Frequency of tiredness / lethargy in last 2 weeks | Instance 0"] = pd.to_numeric(df["Frequency of tiredness / lethargy in last 2 weeks | Instance 0"])
    return df

#IPAQ activity group | Instance 0
def encode_ipaq(df):
    df["IPAQ activity group | Instance 0"].replace(
        {"low": 1,
         "moderate": 2,
         "high": 3
        }, inplace=True
    )
    df["IPAQ activity group | Instance 0"] = pd.to_numeric(df["IPAQ activity group | Instance 0"])
    return df

#Alcohol drinker status | Instance 0
def encode_alc_drinker(df):
    df["Alcohol drinker status | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    return df

# Alcohol intake frequency. | Instance 0
def encode_alc_freq(df):
    df["Alcohol intake frequency. | Instance 0"].replace(
        {
            "Never" : 0,
            "Special occasions only": 1,
            "One to three times a month": 2,
            "Once or twice a week": 6,
            "Three or four times a week": 10,
            "Daily or almost daily": 20,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Alcohol intake frequency. | Instance 0"] = pd.to_numeric(df["Alcohol intake frequency. | Instance 0"])
    return df

# Length of mobile phone use | Instance 0
def encode_mob_length(df):
    df["Length of mobile phone use | Instance 0"].replace(
        {
            "Never used mobile phone at least once per week" : 0,
            "One year or less": 1,
            "Two to four years": 3,
            "Five to eight years": 6,
            "More than eight years": 8,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Length of mobile phone use | Instance 0"] = pd.to_numeric(df["Length of mobile phone use | Instance 0"])
    return df

# Sleep duration | Instance 0
def encode_sleep_dur(df):
    df["Sleep duration | Instance 0"].replace(
        {
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Sleep duration | Instance 0"] = pd.to_numeric(df["Sleep duration | Instance 0"])
    return df

# Smoking status | Instance 0
def encode_smoke_stat(df):
    df["Smoking status | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    return df

#Weekly usage of mobile phone in last 3 months | Instance 0
def encode_mob_use(df):
    df["Weekly usage of mobile phone in last 3 months | Instance 0"].replace(
        {
            "Less than 5mins": 0,
            "5-29 mins": 0.5,
            "30-59 mins": 1,
            "1-3 hours": 3,
            "4-6 hours": 6,
            "More than 6 hours": 8,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Weekly usage of mobile phone in last 3 months | Instance 0"] = pd.to_numeric(df["Weekly usage of mobile phone in last 3 months | Instance 0"])
    return df

# Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0
def encode_blood_clot(df):
    df["Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    return df

# Cancer diagnosed by doctor | Instance 0
def encode_cancer(df):
    df["Cancer diagnosed by doctor | Instance 0"].replace(
        {
            "Yes - you will be asked about this later by an interviewer": 1,
            "No": 0,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Cancer diagnosed by doctor | Instance 0"] = pd.to_numeric(df["Cancer diagnosed by doctor | Instance 0"])
    return df

# Chest pain or discomfort | Instance 0
def encode_chest_pain(df):
    df["Chest pain or discomfort | Instance 0"].replace(
        {
            "Yes": 1,
            "No": 0,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Chest pain or discomfort | Instance 0"] = pd.to_numeric(df["Chest pain or discomfort | Instance 0"])
    return df

# Diabetes diagnosed by doctor | Instance 0
def encode_diabetes(df):
    df["Diabetes diagnosed by doctor | Instance 0"].replace(
        {
            "Yes": 1,
            "No": 0,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Diabetes diagnosed by doctor | Instance 0"] = pd.to_numeric(df["Diabetes diagnosed by doctor | Instance 0"])
    return df

# Fractured/broken bones in last 5 years | Instance 0
def encode_bones(df):
    df["Fractured/broken bones in last 5 years | Instance 0"].replace(
        {
            "Yes": 1,
            "No": 0,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Fractured/broken bones in last 5 years | Instance 0"] = pd.to_numeric(df["Fractured/broken bones in last 5 years | Instance 0"])
    return df

#Mouth/teeth dental problems | Instance 0
def encode_dental(df):
    df["Mouth/teeth dental problems | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    return df

# Other serious medical condition/disability diagnosed by doctor | Instance 0
def encode_other_condition(df):
    df["Other serious medical condition/disability diagnosed by doctor | Instance 0"].replace(
        {
            "Yes - you will be asked about this later by an interviewer": 1,
            "No": 0,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Other serious medical condition/disability diagnosed by doctor | Instance 0"] = pd.to_numeric(df["Other serious medical condition/disability diagnosed by doctor | Instance 0"])
    return df

# Overall health rating | Instance 0
def encode_health_rating(df):
    df["Overall health rating | Instance 0"].replace(
        {
            "Excellent": 4,
            "Good": 3,
            "Fair": 2,
            "Poor": 1,
            "Do not know": np.nan,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Overall health rating | Instance 0"] = pd.to_numeric(df["Overall health rating | Instance 0"])
    return df

#Vascular/heart problems diagnosed by doctor | Instance 0
def encode_heart(df):
    df["Vascular/heart problems diagnosed by doctor | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    return df

# Age started wearing glasses or contact lenses | Instance 0
def encode_glasses_age(df):
    df["Age started wearing glasses or contact lenses | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan,
            "Do not know": np.nan
        }, inplace=True
    )
    df["Age started wearing glasses or contact lenses | Instance 0"] = pd.to_numeric(df["Age started wearing glasses or contact lenses | Instance 0"])
    return df

#Breastfed as a baby | Instance 0
def encode_breastfed(df):
    df["Breastfed as a baby | Instance 0"].replace(
        {
            "Yes": 1,
            "No": 0,
            "Prefer not to answer": np.nan,
            "Do not know": np.nan
        }, inplace=True
    )
    df["Breastfed as a baby | Instance 0"] = pd.to_numeric(df["Breastfed as a baby | Instance 0"])
    return df
    
#Getting up in morning | Instance 0
def encode_getting_up(df):
    df["Getting up in morning | Instance 0"].replace(
        {
            "Not at all easy": 0,
            "Not very easy": 1,
            "Fairly easy": 2, 
            "Very easy":3,
            "Prefer not to answer": np.nan,
            "Do not know": np.nan
        }, inplace=True
    )
    df["Getting up in morning | Instance 0"] = pd.to_numeric(df["Getting up in morning | Instance 0"])
    return df

#How are people in household related to participant | Instance 0
def encode_household_relations(df):
    df["How are people in household related to participant | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan,
        }, inplace=True
    )

    return df

#Number in household | Instance 0
def encode_num_in_household(df):
    df["Number in household | Instance 0"].replace(
        {
            "Prefer not to answer": np.nan,
            "Do not know": np.nan
        }, inplace=True
    )
    df["Number in household | Instance 0"] = pd.to_numeric(df["Number in household | Instance 0"])
    return df

#Wears glasses or contact lenses | Instance 0
def encode_glasses(df):
    df["Wears glasses or contact lenses | Instance 0"].replace(
        {
            "Yes": 1,
            "No": 0,
            "Prefer not to answer": np.nan
        }, inplace=True
    )
    df["Wears glasses or contact lenses | Instance 0"] = pd.to_numeric(df["Wears glasses or contact lenses | Instance 0"])
    return df