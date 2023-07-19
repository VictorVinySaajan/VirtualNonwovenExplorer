import numpy as np
import pandas as pd
import math
from ast import literal_eval
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import sklearn.model_selection as model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

def ConvetToArray(row):
    return np.array(literal_eval(row['CV_Value']))

def GetFeaturesAndLabels(dataframe):
    y = np.array(dataframe['CV_Value'])
    y = np.concatenate(y).ravel()
    y.resize(len(dataframe),7)

    X = dataframe.copy()
    del X['CV_Value']
    del X['RandomSeeds']
    return X,y

def MultiLinearRegression(model, name, X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=101)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = preprocessing.StandardScaler().fit(X_train)

    # Create linear regression object
    LinReg = model

    # Train the model using the training sets
    LinReg.fit(scaler.transform(X_train), y_train)

    # Make predictions using the testing set
    y_pred = LinReg.predict(scaler.transform(X_test))
    
    print("MODEL::::::::", name)
    print("\n")
    print('Coefficients: \n', LinReg.coef_)
    print('Total Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Total Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    result = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print("Total Mean Absolute Percentage Error = ", result, "%")
    print("\n")

    print("Results for individual resolutions \n")
    resolutions = [0.5, 1, 2, 5, 10, 20, 50]
    for i in range(0,7):
        print("Resolution ", resolutions[i])
        print('Mean squared error: %.2f'
          % mean_squared_error(y_test[:,i], y_pred[:,i]))
        print('Coefficient of determination: %.2f'
          % r2_score(y_test[:,i], y_pred[:,i]))
        result = np.mean(np.abs((y_test[:,i] - y_pred[:,i]) / y_test[:,i])) * 100
        print("Mean Absolute Percentage Error = ", result, "%")
        print("\n")
    print("--------------------------------------------")


    
def main():
    dataframe = pd.read_csv("SURRO_output_database_complete_1.csv")
    dataframe['CV_Value'] = dataframe.apply(ConvetToArray, axis = 1)
    X,y = GetFeaturesAndLabels(dataframe)
    model_dict = {'Vanilla':LinearRegression(), 'Ridge':Ridge(), 'Lasso':Lasso(), 'ElasticNet':ElasticNet()}
    for model in model_dict:
        MultiLinearRegression(model_dict[model], model, X, y)
                       
if __name__=='__main__':
    main()