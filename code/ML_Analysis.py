import sys
from os.path import dirname, basename, split, join, abspath
import pandas as pd
from DataPreperation import PrepareData
from RandomForestRegressor import RFRegressor
from RegressionNeuralNetwork import NeuralNet
from BayesianRegression import BayesianRegressor
import pickle

def main():
    #Read the data into pandas dataframe
    dataframe = pd.read_csv(sys.argv[1])
    #metrics of different models to be stored
    metrics = list()

    #Split the data into train and testing
    DataPrepObject = PrepareData(dataframe)
    X_train, X_test, y_train, y_test = DataPrepObject.SplitData()

    #ML Model #1: Random Forest
    RandomForestObject = RFRegressor(X_train, X_test, y_train, y_test)
    RFModel, RFScaler, RFMetrics = RandomForestObject.RandomForestRegression()
    #save the model and scaler
    pickle.dump(RFModel, open('../ML_Models/RandomForestRegressor', 'wb'))
    pickle.dump(RFScaler, open('../ML_Models/RandomForestRegressorScaler', 'wb'))
    metrics.append({'Regressor': "Random Forest Regressor", 'MSE': RFMetrics[0], 'R2_Score': RFMetrics[1], 'MAPE(%)': RFMetrics[2]})

    #ML Model #2: Neural Network
    NeuralNetObject = NeuralNet(X_train, X_test, y_train, y_test)
    NNModel, NNScaler, NNMetrics = NeuralNetObject.NeuralNetworkRegressor()
    #save the model and scaler
    pickle.dump(NNModel, open('../ML_Models/NeuralNetworkRegressor', 'wb'))
    pickle.dump(NNScaler, open('../ML_Models/NeuralNetworkRegressorScaler', 'wb'))
    metrics.append({'Regressor': "Neural Network Regressor", 'MSE': NNMetrics[0], 'R2_Score': NNMetrics[1], 'MAPE(%)': NNMetrics[2]})

    #ML Model #2: Bayesian Ridge
    BayesianRegressorObject = BayesianRegressor(X_train, X_test, y_train, y_test)
    BRMetrics = BayesianRegressorObject.BayesianRegression()
    metrics.append({'Regressor': "Bayesian Regressor", 'MSE': BRMetrics[0], 'R2_Score': BRMetrics[1], 'MAPE(%)': BRMetrics[2]})
    #save the model and scaler
    # pickle.dump(BRModel, open('../ML_Models/BayesianRidgeRegressor', 'wb'))
    # pickle.dump(BRScaler, open('../ML_Models/BayesianRidgeRegressorScaler', 'wb'))

    metrics = pd.DataFrame(metrics)
    metrics.to_csv('../Evaluation/ML_Model_Metrics.csv', index=False)

    #dictionary of models
    ModelDict = {
        RFModel: RFMetrics[2],
        NNModel: NNMetrics[2],
        }
    
    #Get the best model based on MAPE error and save the model to be used by the visualization
    BestModel = min(ModelDict, key=ModelDict.get)
    pickle.dump(BestModel, open('../ML_Models/BestModel_1', 'wb'))


if __name__=='__main__':
    main()