#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Import training data
#read the file w/ pandas into a spreadsheet like format
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#iloc-> locate specifically [: -> all rows, 1:2 -> only look at cols 1,2 ]
#only looks at Open and High cols
training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling
#scale to remove outliar bias
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data struct w/ 60 timesteps and one output
#initialise 2 empty arrays for training data
x_train = []
y_train = []

for i in range(60,1258):
	x_train.append(training_set_scaled[i-60:i, 0]) 
	y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Initialising the RNN
regressor = Sequential()

#Addition of first LSTM layer
regressor.add(LSTM(units = 50, return_seqences = True, input_shape = (x_train.shape[1], 1)))
#Dropout regularisation







