#Stevie Merrill
#captainbillybob23@gmail.com
"""This will contain the estimator that my project will use"""
from sklearn.base import BaseEstimator
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

class Estimator(BaseEstimator):
    """Functional Estimator"""
    ####################API Required
    
    coefficients: tf.Variable = None #a,b,c

    

    def __init__(self,weight=1.0,learning_rate = 0.0005, epochs = 10000):
        self.coefficients = tf.Variable([weight,weight,weight])
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __equation(self,x):
        return self.coefficients[0]*x**2 + self.coefficients[1]*x + self.coefficients[2]
    
    def __calculate_loss(self,y_actual,y_expected):
        return tf.reduce_mean(tf.square(y_actual - y_expected))
    
    def __train(self, x, y_expected, learning_rate):
        with tf.GradientTape() as gt:
            y_output = self.__equation(x)
            loss = self.__calculate_loss(y_output,y_expected)
        
        new_coefficients = gt.gradient(loss, self.coefficients)
        self.coefficients.assign_sub(new_coefficients * learning_rate)
    
    def fit(self,X,y):
        self.is_fitted_ = True
        current_epochs = []
        losses = []
        for epoch in range(self.epochs):
            y_output = self.__equation(X)
            loss = self.__calculate_loss(y_output,y)
            #print(f"Epoch: {epoch}, loss: {loss.numpy()}")
            current_epochs.append(epoch)
            losses.append(loss)
            self.__train(X, y, self.learning_rate)

    def predict(self,X):
        return self.__equation(X)
    
    def get_weight(self):
        return {'a':self.coefficients[0].numpy().item(),'b':self.coefficients[1].numpy().item(),'c':self.coefficients[2].numpy().item()}
    
    def score(self,X,y):
        return r2_score(y, self.predict(X))

    ################## attributes

