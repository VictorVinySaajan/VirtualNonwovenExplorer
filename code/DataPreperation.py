import pandas as pd
import numpy as np
from ast import literal_eval
import math

class PrepareData():
    def __init__(self, dataframe):
            self.database = dataframe
            
    def ConvetToArray(self, row):
        return np.array(literal_eval(row['CV_Value']))

    def GetFeaturesAndLabels(self, dataframe):
        y = np.array(dataframe['CV_Value'])
        y = np.concatenate(y).ravel()
        y.resize(len(dataframe),7)

        X = dataframe.copy()
        del X['CV_Value']
        del X['RandomSeeds']
        del X['group']
        return X,y 
    
    def GetDataframeFromIndexes(self, group, indexes):
        dataframe = list()
        for k in indexes:
            dataframe.append(group.get_group(k))

        frames = list()
        for i in range(0, len(dataframe)):
            frames.append(dataframe[i])
    
        dataframe = pd.concat(frames, ignore_index=True)
        return dataframe
    
    def SplitData(self):
        self.database['CV_Value'] = self.database.apply(self.ConvetToArray, axis = 1)
        
        #group the dataframe to make sure that same rows with different random seed values falls entirely into train or testset
        group = self.database.groupby(['Sigma_1','Sigma_2','A','BeltSpinRatio','SpinPositionsPerMeterInverse'], sort=False)
    
        dataframe_temp = list()
        index = 0
        for i, key in group:
            index+=1
            #annotate similar groups with same indexes
            key = pd.DataFrame(key)
            key['group'] = index
            dataframe_temp.append(key)
           
        #convert the list to dataframe
        frames = list()
        for j in range(0, len(dataframe_temp)):
            frames.append(dataframe_temp[j])
    
        dataframe_temp = pd.concat(frames, ignore_index=True)
        
        #divide the dataframe into train and testsets
        grp = dataframe_temp.groupby('group')
        splitLength = math.ceil(len(grp)*0.8)

        array = np.arange(1, len(grp)+1)
        np.random.shuffle(array)
        train_index = array[:splitLength]
        test_index = array[splitLength:]
        
        #prepare training set and testing set from the indexes
        training_set  = self.GetDataframeFromIndexes(grp, train_index)
        testing_set  = self.GetDataframeFromIndexes(grp, test_index)
        
        X_train, y_train = self.GetFeaturesAndLabels(training_set)
        X_test, y_test = self.GetFeaturesAndLabels(testing_set)
        
        return X_train, X_test, y_train, y_test