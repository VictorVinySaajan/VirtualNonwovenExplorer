import pandas as pd
import numpy as np
from ast import literal_eval
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

class NeuralNet():
    def __init__(self, X_train, X_test, y_train, y_test):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

    def NeuralNetworkRegressor(self):
        scaler = preprocessing.StandardScaler().fit(self.X_train)  
        
        # Build a Neural Network Model for Mutiple Regression 
        NN_model = Sequential()

        # The Input Layer :
        NN_model.add(Dense(128, kernel_initializer='normal',input_dim = self.X_train.shape[1], activation='relu'))

        # The Hidden Layers :
        NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
        NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

        # The Output Layer :
        NN_model.add(Dense(7, kernel_initializer='normal',activation='linear'))

        # Compile the network :
        NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        
        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
        callbacks_list = [checkpoint]
        
        # Fit the model
        NN_model.fit(scaler.transform(self.X_train), self.y_train, epochs=10, batch_size=32, validation_split = 0.2)
        
        y_pred = NN_model.predict(scaler.transform(self.X_test))

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
            result = np.mean(np.abs((self.y_test[:,i] - y_pred[:,i]) / self.y_test[:,i])) * 100
            print("Mean Absolute Percentage Error = ", result, "%")
            print("\n")
        print("--------------------------------------------")
        return NN_model, scaler, metrics