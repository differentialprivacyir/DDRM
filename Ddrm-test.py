import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error
import mmh3
from WrappedServer import WrappedServer
import math

epsilon = 5
clientsCount = 100000
changeRounds = 20
M = 32
clientsValues = np.random.randint(M, size=clientsCount)
clients = [Client(epsilon) for i in range(clientsCount)]
WServer = WrappedServer(M, epsilon)
realF = np.zeros([changeRounds, M])
testMean = 0

for i in range(changeRounds):
    for j in range(clientsCount):
        #Determine if we should change data or not:
        clientsValues[j] = np.random.randint(M)
        testMean += clientsValues[j]
        newValue = np.zeros(M)
        newValue[clientsValues[j]] = 1
        toReport = newValue[j % M]
        toReport = int(toReport)
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
sumOfAllRoundsEstimations = 0

for index, row in enumerate(realF):
    for index2, number in enumerate(row):
        realMean += (number * index2)

for index, row in enumerate(result):
    for index2, number in enumerate(row):
        outputMean += (number * index2)


for number in result[19]:
    sumOfAllRoundsEstimations += number

realMean /= changeRounds
outputMean /= changeRounds


print('Consumed Differential Privacy Budget:', epsilon * changeRounds)
print("Global Mean difference:", abs(realMean - outputMean))
print("Sum of all estimated frequencies:", sumOfAllRoundsEstimations)

# print("Avg Error: %", np.mean(error))
for i in range(changeRounds):
    print(f"Min Squared Error at {i}'th iteration:", mean_squared_error(realF[i], result[i]))




