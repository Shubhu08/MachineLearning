import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = np.asarray(features)
        self.labels = np.asarray(labels)
        
        #raise NotImplementedError

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds the k nearest neighbours in the training set.
        It needs to return a list of labels of these k neighbours. When there is a tie in distance, 
		prioritize examples with a smaller index.
        :param point: List[float]
        :return:  List[int]
        """
        
        distances = np.array([self.distance_function(feature, point) for feature in self.features])
        #print("Distances : ",distances)
        #k_neighbours = np.argpartition(distances, self.k).tolist()[:self.k]
        
        k_neighbours = distances.argsort().tolist()[:self.k]
        
        #print("K_neighbours index : ", self.k ,k_neighbours)
        #print("Labels : ",self.labels)
        #print("Predicted : ",self.labels[k_neighbours])
        return self.labels[k_neighbours]
        
        #raise NotImplementedError
		
	# TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data point.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        predicted = []
        for data_point in features:
            k_neighbors = self.get_k_neighbors(data_point)
            predicted_val = Counter(k_neighbors).most_common(1)[0][0]
            #print("Predicted Val : ",predicted_val)
            predicted = predicted + [predicted_val]
            
        
        return predicted
            
        #raise NotImplementedError	


if __name__ == '__main__':
    print(np.__version__)
