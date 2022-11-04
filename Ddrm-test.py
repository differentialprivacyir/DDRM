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
    realF = np.zeros([changeRounds, M])
    testMean = 0

    numberMean = np.zeros(changeRounds * clientsCount)
    selectedNumbers = np.zeros(changeRounds * clientsCount)

    for i in range(changeRounds):
        clientsValues = dataset[:, i]
        for j in range(clientsCount):
            numberMean[i*clientsCount + j] = clientsValues[j]

            testMean += clientsValues[j]
            newValue = np.zeros(M)
            newValue[clientsValues[j]] = 1
            toReport = int(newValue[j % M])
            if toReport == 1:
                selectedNumbers[i*clientsCount + j] = clientsValues[j]
            else:
                selectedNumbers[i*clientsCount + j] = 0
            [v, h] = clients[j].report(toReport)
            WServer.newValue(v, h, j%M)
            realF[i][j % M] += toReport
        WServer.predicate()

    realF /= (clientsCount/M)
    result = WServer.finish()

    # for index in range(changeRounds):  # calculating errors
    #     error.append((result[index] - realF[index]) / realF[index] * 100)
    #     print(index, "-> Estimated:", result[index], " Real:", realF[index], " Error: %", int(error[-1]))

    realMean = 0
    outputMean = 0

    for index, row in enumerate(realF):
        for index2, number in enumerate(row):
            realMean += (number*(clientsCount/M) * index2)
    realMean /= (clientsCount*changeRounds)

    for index, row in enumerate(result):
        for index2, number in enumerate(row):
            outputMean += (number*(clientsCount/M) * index2)
    outputMean /= (clientsCount*changeRounds)

    # realMean /= changeRounds
    # outputMean /= changeRounds


    print('Consumed Differential Privacy Budget:', epsilon * changeRounds)
    print("Global Mean difference:", abs(realMean - outputMean))
    print("Output Mean is:", outputMean)
    print("Mean of bits is:", realMean)
    print("Mean of generated numbers:", np.mean(numberMean))
    print("Mean of sean values by server:", np.mean(selectedNumbers))

    # print("Avg Error: %", np.mean(error))
    for i in range(changeRounds):
        print(f"Min Squared Error at {i}'th iteration:", mean_squared_error(realF[i], result[i]))




