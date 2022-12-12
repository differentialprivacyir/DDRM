import numpy as np
from Client import Client
from Server import Server
from WrappedClient import WrappeedClient
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
    numberOfBits = math.floor(math.log2(M)) + 1
    clientsValues = dataset[:, day]
    # print(np.mean(dataset))
    # clients = [Client(epsilon) for i in range(clientsCount)]
    # clients = np.ndarray(shape=(clientsCount, numberOfBits), dtype=Client)
    # for i in range(clientsCount):
    #     for j in range(numberOfBits):
    #         clients[i][j] = Client(epsilon)
    WServer = WrappedServer(numberOfBits, epsilon)
    WClient = [WrappeedClient(numberOfBits, epsilon) for i in range(clientsCount)]
    realF = np.zeros([changeRounds, numberOfBits])
    testMean = 0

    numberMean = np.zeros(changeRounds * clientsCount)
    selectedNumbers = np.zeros(changeRounds * clientsCount)

    for i in range(changeRounds):
        clientsValues = dataset[:, i]
        for j in range(clientsCount):
            numberMean[i*clientsCount + j] = clientsValues[j]

            testMean += clientsValues[j]
            [allV, allH] = WClient[j].report(clientsValues[j])
            for k in range(len(allV)):
                WServer.newValue(allV[k], allH[k], k)
            # binaryRepresentation = f'{clientsValues[j]:0{numberOfBits}b}'
            # # newValue = np.zeros(M)
            # # newValue[clientsValues[j]] = 1
            # newValue = [c for c in binaryRepresentation]
            # for k in range(numberOfBits):
            #     toReport = int(newValue[k])
            #     # if toReport == 1:
            #     #     selectedNumbers[i*clientsCount + j] = clientsValues[j]
            #     # else:
            #     #     selectedNumbers[i*clientsCount + j] = 0
            #     [v, h] = clients[j][k].report(toReport)
            #     WServer.newValue(v, h, k)
            #     realF[i][k] += toReport
        WServer.predicate()

    result = WServer.finish()
    print(numberMean)
    # for index in range(changeRounds):  # calculating errors
    #     error.append((result[index] - realF[index]) / realF[index] * 100)
    #     print(index, "-> Estimated:", result[index], " Real:", realF[index], " Error: %", int(error[-1]))

    outputMean = 0

    for index, row in enumerate(result):
        for index2, number in enumerate(row):
            outputMean += (number*(clientsCount) * 2 ** (len(row) - 1 - index2))
    outputMean /= (clientsCount*changeRounds)

    # realMean /= changeRounds
    # outputMean /= changeRounds

    clientsChanges = [WClient[i].howManyChanges() for i in range(clientsCount)]
    maximum = np.max(clientsChanges);

    print('Maximum consumed Differential Privacy Budget:', epsilon * maximum)
    print("Global Mean difference:", abs(np.mean(numberMean) - outputMean))
    print("Output Mean is:", outputMean)
    print("Mean of generated numbers:", np.mean(numberMean))
    print("Mean of sean values by server:", np.mean(selectedNumbers))

    # print("Avg Error: %", np.mean(error))
    for i in range(changeRounds):
        print(f"Min Squared Error at {i}'th iteration:", mean_squared_error(realF[i], result[i]))




