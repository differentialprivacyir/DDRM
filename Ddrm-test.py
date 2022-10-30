import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error
import mmh3
from WrappedServer import WrappedServer
import math
import pandas as pd

# Read dataset from file:
csvContent = pd.read_csv('./uniform.csv')
dataset = csvContent.to_numpy()

for epsilon in range(10):
    epsilon += 1
    clientsCount = len(dataset)
    changeRounds = len(dataset[0])
    day = 0 # The i'th day of processing.
    M = 1000 # The maximum value which can appear.
    clientsValues = dataset[:, day]
    clients = [Client(epsilon) for i in range(clientsCount)]
    WServer = WrappedServer(M, epsilon)
    realMean = np.array([])

    for i in range(changeRounds):
        clientsValues = dataset[:, i]
        realMean = np.append(realMean,clientsValues)
        for j in range(clientsCount):
            newValue = np.zeros(M)
            newValue[clientsValues[j]] = 1
            toReport = int(newValue[j % M])
            [v, h] = clients[j].report(toReport)
            WServer.newValue(v, h, j % M)
        WServer.predicate()

    result = WServer.finish()
    outputMean = np.zeros(changeRounds)

    for i in range(changeRounds):
        for j in range(M):
            outputMean[i] += result[i][j] * (j+1)

    print('Consumed Differential Privacy Budget:', epsilon * changeRounds)
    print("Global Mean difference:", abs(realMean - np.mean(outputMean)))

    for i in range(changeRounds):
        print(f"Min Squared Error at {i}'th iteration:", mean_squared_error(realF[i], result[i]))




