import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    err = None
    
    y_val = np.dot(X, w)
    err = np.mean(np.square(np.subtract(y_val, y)))
    print("Error :",err)
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		
  w = None
  X_T = X.transpose()

  w = np.dot(np.linalg.inv(np.dot(X_T,X)),np.dot(X_T,y))
  #print(w)
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    w = None
    
    X_T = X.transpose()
    X_T_X = np.dot(X_T,X)
    #print("X_T_X :",X_T_X.shape)
    I = np.identity(X_T_X.shape[0])
    #print("I :",I.shape)
    #print("I :",I.dtype)
    #print("lambd :",type(lambd), lambd)
    lambd_I = lambd * I
    X_T_X_lambd = X_T_X + lambd_I
    X_T_X_inv = np.linalg.inv(X_T_X_lambd)
    w = np.dot(X_T_X_inv,np.dot(X_T,y))
    
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    bestlambda = None
    best_mse = float("inf")
    
    for lambda_power in range(-14,1):
        #print("Lambda Power :",lambda_power)
        lambd = 2**(lambda_power)
        w_dash = regularized_linear_regression(Xtrain, ytrain, lambd)
        mse = mean_square_error(w_dash, Xval, yval)
        if mse < best_mse:
            bestlambda = lambd
            best_lambda_power = lambda_power
            best_mse = mse
            
    #print("Best Lambda :", best_lambda_power)
    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    
    X_dash = X
    for power in range(2, p + 1):
        X = np.concatenate((X, np.power(X_dash, power)), axis=1)


    return X
    


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

