# message size and latency and packetloss rate
import numpy as np
import glob
import pandas as pd
from functools import reduce
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
# load the dataset

path = r'./recordData/msgTmout/ack1/'
all_paths = glob.glob(path+'*.csv')
all_csvs = list(map(lambda x: np.loadtxt(x,delimiter=",", skiprows=1, usecols=(7,14,19)), all_paths))
dataset = reduce((lambda x,y: np.concatenate((x,y), axis=0)), all_csvs)
numRecords = dataset.shape[0] # Total number of exp records
numTrains = numRecords-18
paraDim = 1

trainset = dataset[0:numTrains, :]
testset = dataset[numTrains:numRecords, :]
trainX = trainset[:,0]
trainY = trainset[:,1]
testX = testset[:, 0]
testY = testset[:, 1]
modelTitle = 'msgTmout'
print(trainX, trainY)
# define the keras model
model = Sequential()
model.add(Dense(64, activation="sigmoid", input_dim=paraDim))
model.add(Dense(128, activation="tanh"))
model.add(Dense(128, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))


# compile the keras model
model.compile(loss='mae',
            optimizer=optimizers.Adam(lr=0.0001),
            metrics=['mae', 'mse', 'accuracy', 'mape'])
# fit the keras model on the dataset
history = model.fit(trainX, trainY, validation_split=0.25, epochs=5000, batch_size=50)
model.save('./savedModels/'+modelTitle+'/ack1.h5')
# evaluate the keras model

xAxis = np.arange(100,5100,100)
yPredictions = model.predict(xAxis)

fig1 = plt.figure()
plt.scatter(testX, testY)
plt.plot(trainX, trainY)
plt.plot(xAxis,yPredictions)
fig1.savefig('./figures/'+modelTitle+'/ack1/PredictionResult.png')

fig2 = plt.figure()
plt.plot(history.history['loss'], color='#2274A5', alpha=0.8)
plt.plot(history.history['val_loss'], color='#FF5140', linestyle='--', alpha=0.8)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('MAE')
fig2.savefig('./figures/'+modelTitle+'/ack1/lossHistory.png')

