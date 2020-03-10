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

path = r'./msgsize-latency/'
all_paths = glob.glob(path+'*.csv')
all_csvs = list(map(lambda x: np.loadtxt(x,delimiter=",", skiprows=1, usecols=(2,5,6,7,14,19)), all_paths))
dataset = reduce((lambda x,y: np.concatenate((x,y), axis=0)), all_csvs)
numRecords = dataset.shape[0] # Total number of exp records
numTrains = numRecords-5
paraDim = 1

trainset = dataset[0:numTrains, :]
testset = dataset[numTrains:numRecords, :]
trainX = trainset[:,1]
trainY = trainset[:,5]
testX = testset[:, 1]
testY = testset[:, 5]
modelTitle = 'mz_ly'
print(trainX, trainY)
# define the keras model
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=paraDim))
model.add(Dense(128, activation="linear"))
model.add(Dense(1, activation="linear"))

# compile the keras model

model.compile(loss='mse',
            optimizer=optimizers.RMSprop(lr=0.01),
            metrics=['mae', 'mse', 'accuracy', 'mape'])
# fit the keras model on the dataset
history = model.fit(trainX, trainY, validation_split=0.25, epochs=5000, batch_size=10)
model.save('./savedModels/'+modelTitle+'/mszLtcy.h5')
# evaluate the keras model

yPredictions = model.predict(testX)

fig1 = plt.figure()
plt.scatter(testX, testY)
plt.scatter(trainX, trainY)
plt.plot(testX,yPredictions)
fig1.savefig('./figures/'+modelTitle+'/PredictionResult.png')

fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
fig2.savefig('./figures/'+modelTitle+'/lossHistory.png')

