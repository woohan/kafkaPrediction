# Packet loss and batch size
import numpy as np
import glob
import pandas as pd
from functools import reduce
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
# load the dataset

path = r'./recordData/batch-pktloss/ack0/'
all_paths = glob.glob(path+'*.csv')
all_csvs = list(map(lambda x: np.loadtxt(x,delimiter=",", skiprows=1, usecols=(2,6,7,14)), all_paths))
dataset = reduce((lambda x,y: np.concatenate((x,y), axis=0)), all_csvs)
numRecords = dataset.shape[0] # Total number of exp records
numTrains = numRecords-500
paraDim = 2

trainset = dataset[0:numTrains, :]
testset = dataset[numTrains:numRecords, :]
trainX = trainset[:,0:paraDim]
trainY = trainset[:,3]
testX = testset[:, 0:paraDim]
testY = testset[:, 3]
# define the keras model
model = Sequential()
model.add(Dense(200, activation="relu", input_dim=paraDim, kernel_initializer="uniform"))
model.add(Dense(200, activation="tanh"))
model.add(Dense(200, activation="tanh"))
model.add(Dense(64, activation="tanh"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

# compile the keras model
optimizer = optimizers.SGD(lr=0.5)
model.compile(loss='mse',
            optimizer='sgd',
            metrics=['mae', 'mse', 'accuracy', 'mape'])
# fit the keras model on the dataset
history = model.fit(trainX, trainY, validation_split=0.25, epochs=500, batch_size=10)
model.save('./savedModels/b_p_a0/lossRate.h5')
# evaluate the keras model
yPredictions = model.predict(testX)[:,0]

fig1 = plt.figure()
xAxis = testset[:,0]
plt.plot(xAxis,yPredictions)
fig1.savefig('./figures/b_p_a0/PredictionResult.png')

fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('MAE')
fig2.savefig('./figures/b_p_a0/lossHistory.png')

# print(trainX[25],trainY[25],testX[25],testY[25])
