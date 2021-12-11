import pandas as pd
from skopt import BayesSearchCV
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


def score(y_true,y_pred): 
    """ Function to print the metrics of interest of the model """
    mse = mean_squared_error(y_true, y_pred) #set score here and not below if using MSE in GridCV
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    print("MSE is: ", mse)
    print("R2 is: ", r2)
    print("Explained variance is:", ev)
    
def model_tune(model : "SKLearn model", 
               param_grid : dict, 
               X_train : pd.DataFrame, 
               y_train : pd.Series,
               n_iter : int = 30,
               cv : int = 5):
    """
        Uses bayesian optimisation to explore the parameter space and scores each set of parameters using 
        CV. Returns the best model with the best found parameters
    """
    # Tune the model with Bayesian optimisation
    opt = BayesSearchCV(model, param_grid, n_iter=n_iter, cv=cv, verbose=1)
    opt.fit(X_train, y_train)
    # With the following parameter combination being optimal
    print("Best parameter combo:", opt.best_params_)
    # Having the following score
    print("Best validation MSE:", opt.best_score_)
    return opt.best_estimator_