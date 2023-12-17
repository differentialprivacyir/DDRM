import numpy as np
from Client import Client
from Server import Server
from WrappedClient import WrappeedClient
from sklearn.metrics import mean_squared_error
from WrappedServer import WrappedServer
import math
import pandas as pd
from datetime import datetime
import sys
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Read dataset from file:
csvContent = pd.read_csv(f'./hpcDatasets/{sys.argv[2]}.csv')
dataSet = np.transpose(csvContent.to_numpy())

# Overal Algorithm Execution Rounds:
OAER =50
# Number of rounds to get reports by server:
ROUND_CHANGES = 2
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
averageMSE = [[0] * ROUND_CHANGES for i in levels]
averageMAE = [[0] * ROUND_CHANGES for i in levels]
averageME = [[0] * ROUND_CHANGES for i in levels]

for oaer in range(OAER):
    print(f'Start of round {oaer} at:', datetime.now())
    N = int(sys.argv[1]) * 1000
    # Determines how many different value types are available in dataset: 
    # (Each bit is responsible for a separate value)
    DATA_SET_SIZE = 8
    POPULATION_SIZE_AT_EACH_LEVEL = int(N/len(levels))
    clientSelectedLevel = [0] * int(N/len(levels)) + [1] * int(N/len(levels)) + [2] * int(N/len(levels)) + [3] * int(N/len(levels)) + [4] * int(N/len(levels))
    # Prepare to keep results of estimations:
    startRoundTime = time()

    clientsCount = N
    changeRounds = ROUND_CHANGES
    day = 0
    M = 2 ** DATA_SET_SIZE - 1
    numberOfBits = DATA_SET_SIZE
    Wservers = [WrappedServer(numberOfBits, eps) for eps in levels]
    WClient = [WrappeedClient(numberOfBits, clientSelectedLevel[i]) for i in range(clientsCount)]

    realF = np.zeros([changeRounds, numberOfBits])
    testMean = 0
    numberMean = np.zeros(changeRounds * clientsCount)
    selectedNumbers = np.zeros(changeRounds * clientsCount)

    for i in range(ROUND_CHANGES):
        print(f'round {i} started')
        startTimestamp = time()
        for j in range(clientsCount):
            numberMean[i*clientsCount + j] = dataSet[i][j]
            testMean += dataSet[i][j]
            [allV, allH] = WClient[j].report(dataSet[i][j])
            for k in range(len(allV)):
                Wservers[clientSelectedLevel[j]].newValue(allV[k], allH[k], k)
            # print(j, dataSet[i][j], clientSelectedLevel[j])
        endTimestamp = time()
        print(f'Clients reported at {(endTimestamp-startTimestamp)/60} minutes')
        startTimestamp = time()
        for Wserver in Wservers:
            Wserver.predicate()
        endTimestamp = time()
        print(f'Server estimated at {(endTimestamp-startTimestamp)/60} minutes')
        startTimestamp = time()
        
    endRoundTime = time()
    print(f'Round took {(endRoundTime - startRoundTime) / 60} minutes.')
    
    estimations = [0] * len(levels)
    for serverIndex in range(len(Wservers)):
        estimations[serverIndex] = Wservers[serverIndex].finish()
        print(serverIndex, estimations[serverIndex])
    exit(0)

    frequencies = []
    for singleRound in dataSet:
        bitRepresentationOfDataSet = [bin(i)[2:].zfill(DATA_SET_SIZE) for i in singleRound]
        numericalBitRepresentationDataSet = [[int(char) for char in list(data)] \
                                            for data in bitRepresentationOfDataSet]
        frequencies.append(np.sum(numericalBitRepresentationDataSet, axis=0))
    normalized = np.array(frequencies) / N


    for r in range(ROUND_CHANGES):
        print(f'\n\n\n ========================================== \nResults of Round {r}:\n==========================================')
        error = []
        for levelIndex in range(len(levels)):
            print(f'Evaluation for level eps = {levels[levelIndex]}')
            for i, _ in enumerate(normalized[r]):  # calculating errors
                error.append(abs(estimations[levelIndex][r][i] - normalized[r][i]) * 100)
                print("index:", i, "-> Estimated:", estimations[levelIndex][r][i], " Real:", normalized[r][i], " Error: %", int(error[-1]))
            print("Global Mean Square Error:", mean_squared_error(normalized[r], estimations[levelIndex][r]))
            print("Global Mean Absolute Error:", mean_absolute_error(normalized[r], estimations[levelIndex][r]))
            averageMSE[levelIndex][r] = (averageMSE[levelIndex][r] * oaer + mean_squared_error(normalized[r], estimations[levelIndex][r]))/(oaer+1)
            averageMAE[levelIndex][r] = (averageMAE[levelIndex][r] * oaer + mean_absolute_error(normalized[r], estimations[levelIndex][r]))/(oaer+1)

    meanOfRounds = np.mean(dataSet, axis=1)
    print('Real mean of rounds is:', meanOfRounds)
    # for index in range(changeRounds):  # calculating errors
    #     error.append((result[index] - realF[index]) / realF[index] * 100)
    #     print(index, "-> Estimated:", result[index], " Real:", realF[index], " Error: %", int(error[-1]))
    outputMean = []
    for r in range(ROUND_CHANGES):
        outputMean.append([])
        for levelIndex in range(len(levels)):
            estimatedMean = 0
            for index2, number in enumerate(estimations[levelIndex][r]):
                estimatedMean += (number*(POPULATION_SIZE_AT_EACH_LEVEL) * 2 ** (len(estimations[levelIndex][r]) - 1 - index2))
            estimatedMean /= POPULATION_SIZE_AT_EACH_LEVEL
            print(f'Mean Difference at round {r} and level {levels[levelIndex]}:', abs(estimatedMean - meanOfRounds[r]))
            averageME[levelIndex][r] = (averageME[levelIndex][r] * oaer + abs(estimatedMean - meanOfRounds[r]))/(oaer+1)
            outputMean[r].append(estimatedMean)        
    print("Estimated Means are:", outputMean)

print("Results for Averaged MSE:", averageMSE)
print("Results for Averaged MAE:", averageMAE)
print("Results for Averaged ME:", averageME)
