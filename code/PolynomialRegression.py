import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, validation_curve

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

def GetBestDegree(X,y):
    test_set = list()
    validation_set = list()
    train_set = list()
    
    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,test_size=0.40, random_state=42)
    
    for count, degree in enumerate([1,2,3,4,5,6,7,8,9,10,11,12]):
        scaler = preprocessing.StandardScaler().fit(X_train)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

        # Train the model using the training sets
        model.fit(scaler.transform(X_train), y_train)

        # Make predictions using the testing set
        y_pred = model.predict(scaler.transform(X_test))

        # Cross validation to see overfitting
        scores = cross_val_score(model, scaler.transform(X_train), y_train, scoring="neg_mean_squared_error", cv=5)

        train_set.append(mean_squared_error(y_train, model.predict(scaler.transform(X_train))))
        validation_set.append(-scores.mean())
        test_set.append(mean_squared_error(y_test, y_pred))
        
    test_set = np.array(test_set)
    index = np.where(test_set == np.min(test_set))
    return int(index[0]+1)

def PolynomialRegression(X, y, degree):
    print("Polynomial Degree", degree)
    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,test_size=0.40, random_state=42)
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Train the model using the training sets
    model.fit(scaler.transform(X_train), y_train)

    # Make predictions using the testing set
    y_pred = model.predict(scaler.transform(X_test))
    
    print('Total Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Total Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    result = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print("Total Mean Absolute Percentage Error = ", result, "%")
    print("\n")
    
    print("Results for individual resolutions \n")
    resolutions = [0.5, 1, 2, 5, 10, 20, 50]
    for i in range(0,7):
        print("Resolution ", resolutions[i])
        print('Mean squared error: %.2f' % mean_squared_error(y_test[:,i], y_pred[:,i]))
        print('Coefficient of determination: %.2f' % r2_score(y_test[:,i], y_pred[:,i]))
        result = np.mean(np.abs((y_test[:,i] - y_pred[:,i]) / y_test[:,i])) * 100
        print("Mean Absolute Percentage Error = ", result, "%")
        print("\n")
    print("--------------------------------------------")
    
def main():
    dataframe = pd.read_csv("SURRO_output_database_final.csv")
    dataframe['CV_Value'] = dataframe.apply(ConvetToArray, axis = 1)
    X,y = GetFeaturesAndLabels(dataframe)
    degree = GetBestDegree(X,y)
    PolynomialRegression(X, y, degree)

                       
if __name__=='__main__':
    main()