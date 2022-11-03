import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    #print("Real Labels : ",real_labels)
    #print("Predicted Labels : ",predicted_labels)
    #print("Zip : ",list(zip(predicted_labels,real_labels)))
    
    TP, FP, FN = (0,0,0)
    for (pred, real) in zip(predicted_labels,real_labels):
        if (pred,real) == (1,1):
            TP = TP + 1
        elif (pred,real) == (1,0):
            FP = FP + 1
        elif (pred,real) == (0,1):
            FN = FN + 1
    #print("TP, FP, FN : ", TP, FP, FN)
    return TP / (TP + ((FP + FN)/2) )
    
#    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum_of_cubes = 0
        for (x1, x2) in zip(point1, point2):
            sum_of_cubes = sum_of_cubes + abs(x1-x2)**3
                   
        return sum_of_cubes**(1./3)
        
        #raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #print("Point 1 : ", point1)
        #print("Point 2 : ", point2)
        #print("Point pairs : ",list(zip(point1, point2)))
        
        sum_of_sqrs = 0
        for (x1, x2) in zip(point1, point2):
            #print("X1 - X2", x1,x2,x1-x2,abs(x1-x2),abs(x1-x2)**2)
            sum_of_sqrs = sum_of_sqrs + abs(x1-x2)**2
            
        #print("Sum of squares : ",sum_of_sqrs)
        
        #print("Euclidean Distance : ",sum_of_sqrs**(1./2))            
        return sum_of_sqrs**(1./2)
        #raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
       """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
       sum_of_products = 0
       sum_of_x1 = 0
       sum_of_x2 = 0
       
       for (x1, x2) in zip(point1, point2):
           sum_of_products = sum_of_products + (x1*x2)
           sum_of_x1 = sum_of_x1 + (x1**2)
           sum_of_x2 = sum_of_x2 + (x2**2)              
        
       return (1 - sum_of_products / ((sum_of_x1*sum_of_x2)**(1./2))) if ((sum_of_x1*sum_of_x2) != 0) else 1
       #raise NotImplementedError



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        self.best_f1_score = -1.0

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        self.best_f1_score = -1.0
        
        #print("Distance functions : ",distance_funcs)
        
        for distance_function in distance_funcs:  
            #print("distance_function : ",distance_function)
            for k in range(1, min(30, len(x_train)+1) , 2):
                model = KNN(k, distance_funcs[distance_function])
                model.train(x_train, y_train)

                predictions = model.predict(x_val)

                f1_score_value = f1_score(y_val, predictions)
                #print("F1 Score : ",f1_score_value)
                
                if f1_score_value > self.best_f1_score:
                    self.best_k = k
                    self.best_distance_function = distance_function
                    self.best_model = model
                    self.best_f1_score = f1_score_value
                
        print("best_f1_score :",self.best_f1_score)
        print("best_k :",self.best_k)
        print("best_distance_function :",self.best_distance_function)
    
                                        
        
#        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None        
        self.best_f1_score = -1.0


        for scaling_name, scaling_class in scaling_classes.items():
            print(scaling_name)
            scale_obj = scaling_class()          
            normalized_x_train = scale_obj(x_train)          
            normalized_x_val = scale_obj(x_val)            
            
            for distance_function in distance_funcs:
                for k in range(1, min(len(x_train), 30), 2):
                    model = KNN(k, distance_funcs[distance_function])
                    model.train(normalized_x_train, y_train)
    
                    predictions = model.predict(normalized_x_val)
    
                    f1_score_value = f1_score(y_val, predictions)
                    
                    #print("F1 Score : ",f1_score_value)
                    
                    if f1_score_value > self.best_f1_score:
                        self.best_k = k
                        self.best_distance_function = distance_function
                        self.best_model = model
                        self.best_f1_score = f1_score_value
                        self.best_scaler = scaling_name
                        
                        print("best_f1_score :",self.best_f1_score)
                        print("best_k :",self.best_k)
                        print("best_distance_function :",self.best_distance_function)
                        print("best_scaler :",self.best_scaler)
                        
      
        #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        np_features = np.array(features) 
        for i in range(np_features.shape[0]):
            #print("Features : ",np_features[i])
            sum_of_sqrs = 0
            for data in np_features[i]:
                sum_of_sqrs = sum_of_sqrs + data*data
            
            denominator = np.sqrt(sum_of_sqrs)
            np_features[i] = (np_features[i])/denominator if denominator != 0 else np_features[i]
        
        return np_features
        
        #raise NotImplementedError


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        
        np_features = np.array(features) 
        for i in range(np_features.shape[1]):
            min_val = np.min(np_features[:,i])
            max_val = np.max(np_features[:,i])
            np_features[:,i] = (np_features[:,i] - min_val)/(max_val - min_val) if (max_val - min_val) != 0 else 0
        
        #print(np_features)
        return np_features
        
#        raise NotImplementedError
