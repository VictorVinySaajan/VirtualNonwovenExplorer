import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score

class RFRegressor():
    def __init__(self, X_train, X_test, y_train, y_test):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

    def RandomForestRegression(self):
        print("Computing RF Model")
        scaler = preprocessing.StandardScaler().fit(self.X_train)    

        # Fitting Random Forest Regression to the training dataset
        regressor = RandomForestRegressor()
        regressor.fit(scaler.transform(self.X_train), self.y_train)

        # Make predictions using the testing set
        y_pred = regressor.predict(scaler.transform(self.X_test))

        metrics = list()

        print('Total Mean squared error: %.2f' % mean_squared_error(self.y_test, y_pred))
        print('Total Coefficient of determination: %.2f' % r2_score(self.y_test, y_pred))
        MAPE = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        print("Total Mean Absolute Percentage Error = ", MAPE, "%")
        print("\n")

        metrics.append(mean_squared_error(self.y_test, y_pred))
        metrics.append(r2_score(self.y_test, y_pred))
        metrics.append(MAPE)

        print("Results for individual resolutions \n")
        resolutions = [0.5, 1, 2, 5, 10, 20, 50]
        for i in range(0,7):
            print("Resolution ", resolutions[i])
            print('Mean squared error: %.2f' % mean_squared_error(self.y_test[:,i], y_pred[:,i]))
            print('Coefficient of determination: %.2f' % r2_score(self.y_test[:,i], y_pred[:,i]))
            MAPE_Individual = np.mean(np.abs((self.y_test[:,i] - y_pred[:,i]) / self.y_test[:,i])) * 100
            print("Mean Absolute Percentage Error = ", MAPE_Individual, "%")
            print("\n")
        print("--------------------------------------------")
        return regressor, scaler, metrics