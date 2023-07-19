#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy
import pandas
import itertools
import random
import math
import pyDOE

# systemRadom() class
src =random.SystemRandom()

# generates 3 random numbers to use as randomSeeds for reproducibility
def GenerateRandomSeeds(row):
    a = src.randint(0, 2**31-1)
    b = src.randint(0, 2**31-1)
    c = src.randint(0, 2**31-1)
    return [a, b, c]

def MapToLogSpace(dimension, lowerBound, upperBound):
    return numpy.exp(numpy.log(upperBound) * dimension + numpy.log(lowerBound) * (1 - dimension))

def GetLatinHypercubeSamples(dimensions, sampleSize):
    dataframe = pandas.DataFrame(columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", "SpinPositionsPerMeterInverse"])
    samples = pyDOE.lhs(dimensions, samples=sampleSize)
    samples[:, 0] = MapToLogSpace(samples[:, 0], 1, 50)
    samples[:, 1] = MapToLogSpace(samples[:, 1], 1, 50)
    samples[:, 2] = MapToLogSpace(samples[:, 2], 1, 50)
    samples[:, 3] = MapToLogSpace(samples[:, 3], 0.01, 0.25)
    samples[:, 4] = MapToLogSpace(samples[:, 4], 200, 10000)
    dataframe['Sigma_1'] = samples[:, 0]
    dataframe['Sigma_2'] = samples[:, 1]
    dataframe['A'] = samples[:, 2]
    dataframe['BeltSpinRatio'] = samples[:, 3]
    dataframe['SpinPositionsPerMeterInverse'] = samples[:, 4]
    return dataframe

def getDiscreteSamples():
    sigma_1 =  [1, 2, 5, 10, 22, 35, 50]
    sigma_2 = [1, 2, 5, 10, 22, 35, 50]
    A = [1, 2, 5, 10, 22, 35, 50]
    beltSpinRatio = [0.01, 0.02, 0.05, 0.10, 0.18, 0.25]
    spinPositionsPerMeterInverse = [200, 500, 1000, 2000, 5000, 10000]
    
    List = [sigma_1, sigma_2, A, beltSpinRatio, spinPositionsPerMeterInverse]
    dataframe = pandas.DataFrame(list(itertools.product(*List)), columns = ["Sigma_1", "Sigma_2", "A", "BeltSpinRatio", 
                                                                            "SpinPositionsPerMeterInverse"])
    return dataframe
    

# creates a database with the input parameters to the SURRO software
def CreateDataBase():
    df1 = getDiscreteSamples()
    df2 = GetLatinHypercubeSamples(5, 50000)
    
    frames = [df1, df2]
    dataframe = pandas.concat(frames, ignore_index=True)
    
    dataframe = dataframe.loc[dataframe.index.repeat(5)].reset_index(drop=True)
    dataframe['RandomSeeds'] = dataframe.apply(GenerateRandomSeeds, axis=1)
    dataframe.to_csv('SURRO_Complete_input_database.csv', index=False)
    
def main():
    CreateDataBase()

if __name__=='__main__':
    main()


# In[ ]:




