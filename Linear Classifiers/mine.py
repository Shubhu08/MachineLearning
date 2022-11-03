import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
        
    mod_X = np.insert(X, 0, 1, axis=1)
    mod_w = np.insert(w, 0, b, axis=0)
    mod_y = np.where(y == 0, -1, 1)

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        #import logging
        
        #logging.basicConfig(filename='log_filename.txt', level=#logging.debug, format='%(asctime)s - %(levelname)s - %(message)s')

        coeff = step_size / N
        ##logging.debug("binary train perceptron coeff : " + str(coeff))

        for i in range(max_iterations):
            pred_y = np.dot(mod_X,mod_w)
            pred_y = np.where(pred_y < 0, -1, 1)
            y_wT_x = mod_y * pred_y
            mis_class = np.where(y_wT_x <= 0 , 1, 0)
            
            ##logging.debug("binary train perceptron mis_class : " + str(mis_class))
            
            ##logging.debug("binary train perceptron mis_class * y : " + str(mis_class * mod_y))
            
            multiplier = np.dot(mis_class * mod_y, mod_X)
            
            ##logging.debug("binary train perceptron (coeff * multiplier) : " + str((coeff * multiplier)))
            
            mod_w = mod_w + (coeff * multiplier)
        
        ##logging.debug("binary train weights : " + str(mod_w))

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        
        #import logging
        
        #logging.basicConfig(filename='log_filename.txt', level=#logging.debug, format='%(asctime)s - %(levelname)s - %(message)s')

        
        coeff = step_size / N
        for i in range(max_iterations):
            pred_y = np.dot(mod_X,mod_w)
            #pred_y = np.where(pred_y < 0, -1, 1)
            y_wT_x = mod_y * pred_y
            #sig_arg = np.where(y_wT_x < 0 , 1, 0)
            sigmoid_value = sigmoid(y_wT_x * -1)
            multiplier = np.dot(sigmoid_value * mod_y, mod_X)

            mod_w = mod_w + (coeff * multiplier)
            
        ##logging.debug("binary train logistic weights : " + str(mod_w))
    else:
        raise "Undefined loss function."


    b = mod_w[0]
    w = np.delete(mod_w, 0)
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value =  1 / (1 + np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    mod_X = np.insert(X, 0, 1, axis=1)
    mod_w = np.insert(w, 0, b, axis=0)
    
    y = np.dot(mod_X,mod_w)
    preds = np.where(y < 0, 0, 1)

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    mod_X = np.insert(X, 0, 1, axis=1)
    mod_W = np.insert(w, 0, b, axis=1)
    
    
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    import logging
    
    logging.basicConfig(filename='log_filename.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    #logging.debug("y" + str(y))

    if gd_type == "sgd":

        logging.debug("sgd")
        #logging.debug("sgd C : " + str(C))
        #logging.debug("sgd D : " + str(D))
        #logging.debug("sgd mod_W shape : " + str(mod_W.shape))
        ##logging.debug("sgd step_size : " + str(step_size))
        
        ##logging.debug("sgd mod_X.shape : " + str(mod_X.shape))
        ##logging.debug("sgd mod_X.values : " + str(mod_X))
        ##logging.debug("sgd mod_W.shape : " + str(mod_W.shape))
        ##logging.debug("sgd mod_W.values : " + str(mod_W))

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################		
            
            
            xn = mod_X[n]
                        
            ##logging.debug("sgd xn.shape : " + str(xn.shape))
            ##logging.debug("sgd xn.value : " + str(xn))
            ##logging.debug("sgd mod_W.shape : " + str(mod_W.shape))
            
            #W_xn = np.dot(mod_W, mod_X[n])
            W_xn = np.dot(mod_X[n], mod_W.T)
            
            ##logging.debug("sgd W_xn.shape : " + str(W_xn.shape))
            sub = W_xn - np.amax(W_xn)
            num = np.exp(sub)
            #logging.debug("sgd num[0] : " + str(num[0]))
            den = np.sum(num)
            #logging.debug("sgd den : " + str(den))
            
            softmax =  (num / den)  
            #softmax = [i / den for i in num]
            
            softmax[y[n]] = softmax[y[n]] - 1
            
            #softmax = np.array([softmax])
            
            ##logging.debug("sgd y[n] : " + str(y[n]))
            ##logging.debug("sgd softmax : " + str(softmax))
            ##logging.debug("sgd softmax shape : " + str(softmax.T.shape))
            ##logging.debug("sgd mox_X shape : " + str(mod_X[n].shape))
            #logging.debug("sgd mod_W shape : " + str(mod_W.shape))
            ##logging.debug("sgd xn : " + str(xn))
        
            ##logging.debug("sgd xn.value : " + str(xn))
            sgd = np.dot(softmax[:,None], mod_X[n][None,:])
            #sgd = np.matmul(np.array([softmax]).T, np.array([xn]))
            ##logging.debug("sgd sgd.shape : " + str(sgd.shape))
            #logging.debug("sgd sgd.values : " + str(sgd))
            
            
            mod_W = mod_W - step_size * sgd
            
        #logging.debug("sgd mod_W : " + str(mod_W))
        

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        logging.debug("gd")
        correct_pred = np.eye(C)[y]
        #logging.debug("correct_pred" + str(correct_pred))
        #logging.debug("gd correct_pred.values : "+ str(correct_pred))
        #logging.debug("gd C : " + str(C))
        #logging.debug("gd D : " + str(D))
        #logging.debug("gd mod_W.shape : " + str(mod_W.shape))
        for _ in range(max_iterations):
            
            #logging.debug("gd mod_X.shape : " + str(mod_X.shape))
            #logging.debug("gd mod_X.values : " + str(mod_X))
            
            #logging.debug("gd mod_W.shape : " + str(mod_W.shape))
            #logging.debug("gd mod_W.values : " + str(mod_W))            

            W_XT = np.dot(mod_W, mod_X.T)
            #logging.debug("gd W_XT.shape : " + str(W_XT.shape))
            
            
            #logging.debug("gd correct_pred.shape : " + str(correct_pred.shape))
            
            sub = W_XT - np.amax(W_XT)
            num = np.exp(sub)
            #logging.debug("gd W_XT.shape : " + str(W_XT.shape))
            #logging.debug("gd W_XT.values : " + str(W_XT))
            
            den = np.sum(num, axis=0)
            #logging.debug("gd num.shape : " + str(num.shape))
            #logging.debug("gd den.shape : " + str(den.shape))
            #logging.debug("gd den.values : " + str(den))
            
            softmax =  (num / den)
            
            #logging.debug("gd softmax.value : " + str(softmax))
            
            softmax = softmax - correct_pred.T
            
            #logging.debug("gd softmax.shape : " + str(softmax.shape))
            #logging.debug("gd softmax.value : " + str(softmax))
            
            #logging.debug("gd mod_X.shape : " + str(mod_X.shape))
            #logging.debug("gd mod_W.shape : " + str(mod_W.shape))
            gd = np.dot(softmax, mod_X)
            
            #logging.debug("gd gd.shape : " + str(gd.shape))
            
            mod_W = mod_W - (step_size/N * gd)
            
            #logging.debug("gd loop mod_W : " + str(mod_W))
            
            #if(np.isnan(mod_W[0][0])):
            #    logging.debug("gd num.values : " + str(num))
            #    logging.debug("gd den.values : " + str(den))
            #    logging.debug("gd softmax.value : " + str(softmax))
            #    logging.debug("gd mod_X.value : " + str(mod_X))
            #    logging.debug("gd gd.value : " + str(gd))
                
            
        #logging.debug("gd mod_W : " + str(mod_W))
        
    else:
        raise "Undefined algorithm."
    
    logging.debug("mod_W : " + str(mod_W))
    b = mod_W[:,0]
    w = np.delete(mod_W, 0, axis=1)
    
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    mod_X = np.insert(X, 0, 1, axis=1)
    mod_W = np.insert(w, 0, b, axis=1)
    
#    import logging
    
#    logging.basicConfig(filename='log_filename.txt', level=#logging.debug, format='%(asctime)s - %(levelname)s - %(message)s')
#    #logging.debug(mod_X.shape)
#    #logging.debug(mod_W.shape)

    
    probs = np.matmul(mod_W, mod_X.T)
    preds = np.argmax(probs, axis=0)
#    #logging.debug(probs.shape)
#    #logging.debug(preds.shape)
    
    assert preds.shape == (N,)
    return preds




        