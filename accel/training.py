import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


def score(y_true,y_pred): 
    """ Function to print the metrics of interest of the model """
    mse = mean_squared_error(y_true, y_pred) #set score here and not below if using MSE in GridCV
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    print("MSE is: ", mse)
    print("R2 is: ", r2)
    print("Explained variance is:", ev)
    return r2_score, mean_squared_error, explained_variance_score
    
def model_tune(model : "SKLearn model", 
               param_grid : dict, 
               X_train : pd.DataFrame, 
               y_train : pd.Series,
               X_val : pd.DataFrame, 
               y_val : pd.Series,
               n_iter : int = 30):
    """
        Uses bayesian optimisation to explore the parameter space and scores each set of parameters using 
        CV. Returns the best model with the best found parameters
    """
    
    X = np.vstack((X_train, X_val))
    test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_val.shape[0])]
    y = np.concatenate([y_train, y_val])
    ps = PredefinedSplit(test_fold)
    
    # Tune the model with Bayesian optimisation
    opt = BayesSearchCV(model, param_grid, n_iter=n_iter, cv=ps, verbose=1, refit=False)
    opt.fit(X, y)
    
    # With the following parameter combination being optimal
    print("Best parameter combo:", opt.best_params_)
    # Having the following score
    print("Best validation MSE:", opt.best_score_)
    return opt
