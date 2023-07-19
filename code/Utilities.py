import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import pickle
import pyDOE
import itertools

class MachineLearningUtils():
    def __init__(self):
        #Load the ML model and it's scaler for analysis
        self.Best_model = pickle.load(open('../ML_Models/BestModel', 'rb'))
        self.scaler = pickle.load(open('../ML_Models/NeuralNetworkRegressorScaler', 'rb'))
        self.CliModel = pickle.load(open('../ML_Models/RadomForestForCli', 'rb'))

    def getBestModel(self):
        return self.Best_model

    def getBestScaler(self):
        return self.scaler

    def getPredictions(self, input):
        test_array = np.array(input)
        test_array = test_array.reshape(1, -1)
        predictions = self.Best_model.predict(self.scaler.transform(test_array))
        predictions = np.concatenate(predictions).ravel()
        return predictions

    def CliPrediction(self, dataframe):
        test_array = np.array(dataframe)
        test_array = test_array.reshape(1, -1)
        return self.CliModel.predict(test_array)

    def getPredictionList(self, inputList):
        predictionList = []
        for i in inputList:
            test_array = np.array(i)
            test_array = test_array.reshape(1, -1)
            predictions = self.Best_model.predict(self.scaler.transform(test_array))
            predictions = np.concatenate(predictions).ravel()
            predictionList.append(predictions)
        return predictionList

    def getCliPredictionList(self, inputList):
        predictionList = []
        for i in inputList:
            test_array = np.array(i)
            test_array = test_array.reshape(1, -1)
            predictions = self.CliModel.predict(test_array)
            predictions = predictions[0]
            predictionList.append(predictions)
        return predictionList

class DataPreperationUtils():
    def __init__(self):
        self.DataFrame = pd.read_csv('../data/Database.csv')
        self.MLUtils = MachineLearningUtils()

    def MapToLinSpace(self, dimension, lowerBound, upperBound):
        return np.multiply(dimension, (upperBound-lowerBound)) + lowerBound

    def GetLatinHypercubeSamples(self, dimensions, sampleSize, lowerBound, upperBound):
        samples = pyDOE.lhs(dimensions, samples=sampleSize)
        samples = self.MapToLinSpace(samples, lowerBound, upperBound)
        samples = np.concatenate(samples).ravel()
        samples = np.sort(samples)
        return samples

    def GetLatinHypercubeSamplesForAllDim(self, sampleSize):
        dataframe = pd.DataFrame(columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", "SpinPositionsPerMeterInverse"])
        samples = pyDOE.lhs(5, samples=sampleSize)
        samples[:, 0] = self.MapToLinSpace(samples[:, 0], 1, 50)
        samples[:, 1] = self.MapToLinSpace(samples[:, 1], 1, 50)
        samples[:, 2] = self.MapToLinSpace(samples[:, 2], 1, 50)
        samples[:, 3] = self.MapToLinSpace(samples[:, 3], 0.01, 0.25)
        samples[:, 4] = self.MapToLinSpace(samples[:, 4], 200, 10000)
        dataframe['Sigma_1'] = samples[:, 0]
        dataframe['Sigma_2'] = samples[:, 1]
        dataframe['A'] = samples[:, 2]
        dataframe['BeltSpinRatio'] = samples[:, 3]
        dataframe['SpinPositionsPerMeterInverse'] = samples[:, 4]
        return dataframe

    def PerformPCA(self, feature_set):
        # feature_set = preprocessing.StandardScaler().fit_transform(feature_set)  
        pc = PCA(n_components=2)
        return pc.fit_transform(feature_set)

    def PerformGMMClustering(self, data):
        gmm = GaussianMixture(n_components=5)
        gmm.fit(data)
        return gmm.predict(data)

    def getCheckBoxValues(self, val, currValue, lowerBound, upperBound, numberOfSamples):
        if val == 0:
            return np.full(numberOfSamples, currValue)
        else:
            samples =  self.GetLatinHypercubeSamples(1,numberOfSamples, lowerBound, upperBound)
            np.random.shuffle(samples)
            return samples

    def getFullRangePredictions(self, colName, val1, val2, val3, sampleSize, resolution, r1_min, r1_max, r2_min, r2_max):
        Sigma_1, Sigma_2, A, BeltSpinRatio, SpinPositionsPerMeterInverse, col1Name, col2Name =[],[],[],[],[],[],[]
        if colName == 'S1S2':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = col1Name
            Sigma_2 = col2Name
            A = [val1]
            BeltSpinRatio = [val2]
            SpinPositionsPerMeterInverse = [val3]

        if colName == 'S1A':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = col1Name
            Sigma_2 = [val1]
            A = col2Name
            BeltSpinRatio = [val2]
            SpinPositionsPerMeterInverse = [val3]

        if colName == 'S1BSR':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = col1Name
            Sigma_2 = [val1]
            A = [val2]
            BeltSpinRatio = col2Name
            SpinPositionsPerMeterInverse = [val3]

        if colName == 'S1SPM':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = col1Name
            Sigma_2 = [val1]
            A = [val2]
            BeltSpinRatio = [val3]
            SpinPositionsPerMeterInverse = col2Name

        if colName == 'S2A':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = [val1]
            Sigma_2 = col1Name
            A = col2Name
            BeltSpinRatio = [val2]
            SpinPositionsPerMeterInverse = [val3]

        if colName == 'S2BSR':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = [val1]
            Sigma_2 = col1Name
            A = [val2]
            BeltSpinRatio = col2Name
            SpinPositionsPerMeterInverse = [val3]

        if colName == 'S2SPM':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = [val1]
            Sigma_2 = col1Name
            A = [val2]
            BeltSpinRatio = [val3]
            SpinPositionsPerMeterInverse = col2Name

        if colName == 'ABSR':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = [val1]
            Sigma_2 = [val2]
            A = col1Name
            BeltSpinRatio = col2Name
            SpinPositionsPerMeterInverse = [val3]

        if colName == 'ASPM':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = [val1]
            Sigma_2 = [val2]
            A = col1Name
            BeltSpinRatio = [val3]
            SpinPositionsPerMeterInverse = col2Name

        if colName == 'BSRSPM':
            col1Name = self.GetLatinHypercubeSamples(1,sampleSize,r1_min,r1_max)
            col2Name = self.GetLatinHypercubeSamples(1,sampleSize,r2_min,r2_max)
            Sigma_1 = [val1]
            Sigma_2 = [val2]
            A = [val3]
            BeltSpinRatio = col1Name
            SpinPositionsPerMeterInverse = col2Name

        # fullRangeInput = fullRangeInput.drop_duplicates()
        # fullRangeInput.to_csv("fullRangeInput.csv", index= False)

        model = self.MLUtils.getBestModel()
        scaler = self.MLUtils.getBestScaler()

        List = [Sigma_1, Sigma_2, A, BeltSpinRatio, SpinPositionsPerMeterInverse]
        dataframe = pd.DataFrame(list(itertools.product(*List)), columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", 
                                                                            "SpinPositionsPerMeterInverse"])

        #get the predictions for whole range of the given input value keeping others constant
        if resolution == 7:
            predictions = self.MLUtils.CliModel.predict(dataframe)
            predictions = predictions.reshape(25,25)
            fullRangePredictions = pd.DataFrame(data=predictions,index=col1Name,columns=col2Name)
            return fullRangePredictions
        else:
            predictions = model.predict(scaler.transform(dataframe))
            # 0.5 resolutions
            pred = predictions[:,resolution]
            pred = pred.reshape(25,25)
        
            # column = [0.5,1,2,5,10,20,50]

            fullRangePredictions = pd.DataFrame(data=pred,index=col1Name,columns=col2Name)
            return fullRangePredictions


    def getOutputSpacePrediction(self, dataframe):
        return self.MLUtils.getBestModel().predict(self.MLUtils.getBestScaler().transform(dataframe))

    def getClusterData(self):
        dataframe = self.GetLatinHypercubeSamplesForAllDim(10000)
        y = self.getOutputSpacePrediction(dataframe)
        principle_components_op = self.PerformPCA(y)
        labels = self.PerformGMMClustering(principle_components_op)

        principle_components_ip = self.PerformPCA(dataframe)

        df_temp_op = pd.DataFrame()
        df_temp_op['pc_1'] = principle_components_op[:, 0]
        df_temp_op['pc_2'] = principle_components_op[:, 1]
        df_temp_op['label'] = labels

        df_temp_ip = pd.DataFrame()
        df_temp_ip['pc_1'] = principle_components_ip[:, 0]
        df_temp_ip['pc_2'] = principle_components_ip[:, 1]
        df_temp_ip['label'] = labels

        grp_op = df_temp_op.groupby('label')
        grp_ip = df_temp_ip.groupby('label')

        clusterOutputList = []
        clusterInputputList = []
        mean = []
        df_temp = pd.DataFrame()
        df_temp['Res_0.5'] = y[:,0]
        df_temp['Res_1'] = y[:,1]
        df_temp['Res_2'] = y[:,2]
        df_temp['Res_5'] = y[:,3]
        df_temp['Res_10'] = y[:,4]
        df_temp['Res_20'] = y[:,5]
        df_temp['Res_50'] = y[:,6]
        df_temp['label'] = labels

        grp = df_temp.groupby('label')

        mean_ip = []
        dataframe['label'] = labels
        group_ip= dataframe.groupby('label')

        for i,key in group_ip:
            dataframe = pd.DataFrame(key)
            mean_ip.append(dataframe)

        for i,key in grp:
            dataframe = pd.DataFrame(key)
            mean.append(list(dataframe.mean()))

        for i,key in grp_op:
            dataframe = pd.DataFrame(key)
            clusterOutputList.append(dataframe)

        for i,key in grp_ip:
            dataframe = pd.DataFrame(key)
            clusterInputputList.append(dataframe)

        return clusterInputputList, clusterOutputList, mean, mean_ip

    def getDataFrame(self, columnName, val1, val2, val3, val4, sampleSize):
        Sigma_1, Sigma_2, A, BeltSpinRatio, SpinPositionsPerMeterInverse, colName =[],[],[],[],[],[]
        if columnName == 'Sigma_1':
            colName = self.GetLatinHypercubeSamples(1,sampleSize,1,50)
            Sigma_1 = colName
            Sigma_2 = [val1]
            A = [val2]
            BeltSpinRatio = [val3]
            SpinPositionsPerMeterInverse = [val4]

        if columnName == 'Sigma_2':
            colName = self.GetLatinHypercubeSamples(1,sampleSize,1,50)
            Sigma_1 = [val1]
            Sigma_2 = colName
            A = [val2]
            BeltSpinRatio = [val3]
            SpinPositionsPerMeterInverse = [val4]

        if columnName == 'A':
            colName = self.GetLatinHypercubeSamples(1,sampleSize,1,50)
            Sigma_1 = [val1]
            Sigma_2 = [val2]
            A = colName
            BeltSpinRatio = [val3]
            SpinPositionsPerMeterInverse = [val4]

        if columnName == 'BeltSpinRatio':
            colName = self.GetLatinHypercubeSamples(1,sampleSize,0.01,0.25)
            Sigma_1 = [val1]
            Sigma_2 = [val2]
            A = [val3]
            BeltSpinRatio = colName
            SpinPositionsPerMeterInverse = [val4]

        if columnName == 'SpinPositionsPerMeterInverse':
            colName = self.GetLatinHypercubeSamples(1,sampleSize,200,10000)
            Sigma_1 = [val1]
            Sigma_2 = [val2]
            A = [val3]
            BeltSpinRatio = [val4]
            SpinPositionsPerMeterInverse = colName

        List = [Sigma_1, Sigma_2, A, BeltSpinRatio, SpinPositionsPerMeterInverse]
        dataframe = pd.DataFrame(list(itertools.product(*List)), columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", 
                                                                            "SpinPositionsPerMeterInverse"])
        return dataframe

    def getPredictions(self, dataframe):
        model = self.MLUtils.getBestModel()
        scaler = self.MLUtils.getBestScaler()

        predictions =  model.predict(scaler.transform(dataframe))

        return predictions

    def getCliPredictions(self, dataframe):
        return self.MLUtils.CliModel.predict(dataframe)

    def getDerivativeDataFrame(self, columnName, val1, val2, val3, val4, resolution, sampleSize):
        dataframe = self.getDataFrame(columnName, val1, val2, val3, val4, sampleSize)

        dataframe1 = dataframe.copy()
        dataframe2 = dataframe.copy()

        dataframe1[columnName] = dataframe1[columnName] + 0.000001
        dataframe2[columnName] = dataframe2[columnName] - 0.000001

        #calculate the central finite difference as approximation for partial derivative
        prediction1 = self.getPredictions(dataframe1)
        prediction2 = self.getPredictions(dataframe2)

        prediction = prediction1 - prediction2
        prediction = prediction / (2* 0.000001)

        dfRes = pd.DataFrame()

        dfRes[columnName] = dataframe[columnName]

        if resolution <= 6:
            dfRes['Derivative'] = prediction[:,resolution]
        else:
            dfRes['Derivative_half'] = prediction[:,0]
            dfRes['Derivative_1'] = prediction[:,1]
            dfRes['Derivative_2'] = prediction[:,2]
            dfRes['Derivative_5'] = prediction[:,3]
            dfRes['Derivative_10'] = prediction[:,4]
            dfRes['Derivative_20'] = prediction[:,5]
            dfRes['Derivative_50'] = prediction[:,6]

        # df1 = pd.DataFrame()
        # df2 = pd.DataFrame()
        # df3 = pd.DataFrame()
        # df4 = pd.DataFrame()
        # df5 = pd.DataFrame()
        # df6 = pd.DataFrame()
        # df7 = pd.DataFrame()

        # df1[columnName] = dataframe[columnName]
        # df1['Resolution'] = 0.5
        # df1['Derivative'] = prediction[:,0]

        # df2[columnName] = dataframe[columnName]
        # df2['Resolution'] = 1
        # df2['Derivative'] = prediction[:,1]

        # df3[columnName] = dataframe[columnName]
        # df3['Resolution'] = 2
        # df3['Derivative'] = prediction[:,2]

        # df4[columnName] = dataframe[columnName]
        # df4['Resolution'] = 5
        # df4['Derivative'] = prediction[:,3]

        # df5[columnName] = dataframe[columnName]
        # df5['Resolution'] = 10
        # df5['Derivative'] = prediction[:,4]

        # df6[columnName] = dataframe[columnName]
        # df6['Resolution'] = 20
        # df6['Derivative'] = prediction[:,5]

        # df7[columnName] = dataframe[columnName]
        # df7['Resolution'] = 50
        # df7['Derivative'] = prediction[:,6]

        # frames = [df1, df2, df3, df4, df5, df6, df7]
        # result = pd.concat(frames)
        # result['Size'] = np.full(7 * sampleSize ,10)

        return dfRes
