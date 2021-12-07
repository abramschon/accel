import pandas as pd

def load_xy(file_path : str, 
            x_names : [str, ...],
            y_name : str
           ):
    data = pd.read_csv(file_path)
    X = data[x_names]
    y = data[y_name]
    return X, y

def load_data(file_path : str):
    return pd.read_csv(file_path)
