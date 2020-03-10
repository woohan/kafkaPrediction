# Packet loss and batch size ack1 ---- '+modelTitle+'
import numpy as np
import glob
import pandas as pd
from functools import reduce
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
# load the dataset
# dataset = loadtxt("1027.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7,11))
path = r'./recordData/batch-pktloss/ack1/'
all_paths = glob.glob(path+'*.csv')
all_csvs = list(map(lambda x: np.loadtxt(x,delimiter=",", skiprows=1, usecols=(2,6,7,14,18)), all_paths))

dataset2 = reduce((lambda x,y: np.concatenate((x,y), axis=0)), all_csvs)
paraDim = 2


trainset = dataset2[0:4000, :]
testset = dataset2[4000:4500, :]
trainX = trainset[:,0:paraDim]
trainY = trainset[:,4]
testX = testset[:, 0:paraDim]
testY = testset[:, 4]
# define the keras model
model = Sequential()
model.add(Dense(200, activation="relu", input_dim=paraDim, kernel_initializer="uniform"))
model.add(Dense(200, activation="tanh"))
model.add(Dense(200, activation="tanh"))
model.add(Dense(64, activation="tanh"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

modelTitle = 'b_d_a1'
# compile the keras model
optimizer = optimizers.SGD(lr=0.5)
model.compile(loss='mse',
            optimizer='sgd',
            metrics=['mae', 'mse', 'accuracy', 'mape'])
# fit the keras model on the dataset
history = model.fit(trainX, trainY, validation_split=0.25, epochs=1000, batch_size=10)
model.save('./savedModels/'+modelTitle+'/dupliRate.h5')
# evaluate the keras model
yPredictions = model.predict(testX)[:,0]

fig1 = plt.figure()
xAxis = testset[:,0]
plt.plot(xAxis,yPredictions)
fig1.savefig('./figures/'+modelTitle+'/PredictionResult.png')

fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
fig2.savefig('./figures/'+modelTitle+'/lossHistory.png')

# print(trainX[25],trainY[25],testX[25],testY[25])
