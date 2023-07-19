import pandas as pd
import numpy as np
from ast import literal_eval
import sklearn.model_selection as model_selection
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class BayesianRegressor():
    def __init__(self, X_train, X_test, y_train, y_test):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

    def BayesianRegression(self):
        scaler = preprocessing.StandardScaler().fit(self.X_train)    

        model = make_pipeline(PolynomialFeatures(7), BayesianRidge())

        # Train the model using the training sets
        resolutions = [0.5, 1, 2, 5, 10, 20, 50]
        mse = list()
        rs = list()
        mape = list()
        for i in range(0,7):
            model.fit(scaler.transform(self.X_train), self.y_train[:,i])

            y_pred = model.predict(scaler.transform(self.X_test))

            print('Total Mean squared error: %.2f' % mean_squared_error(self.y_test[:,i], y_pred))
            print('Total Coefficient of determination: %.2f' % r2_score(self.y_test[:,i], y_pred))
            result = np.mean(np.abs((self.y_test[:,i] - y_pred) / self.y_test[:,i])) * 100
            print("Total Mean Absolute Percentage Error = ", result, "%")
            y_mean, y_std = model.predict(scaler.transform(self.X_test), return_std=True)
            print("mean",y_mean)
            print("std",y_std)
            print("\n")
            mse.append(mean_squared_error(self.y_test[:,i], y_pred))
            rs.append(r2_score(self.y_test[:,i], y_pred))
            mape.append(result)

        mse = np.array(mse)
        rs = np.array(rs)
        mape = np.array(mape)
        return [mse.mean(), rs.mean(), mape.mean()]