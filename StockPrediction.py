#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

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
	x_train.append(training_set_scaled[]) 
	y_train