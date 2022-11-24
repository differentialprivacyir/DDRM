import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error
import mmh3
from WrappedServer import WrappedServer
import math

epsilon = 1
clientsCount = 5000
changeRounds = 20
M = 1000
clientsValues = np.random.randint(M, size=clientsCount)
#clients = [Client(epsilon) for i in range(clientsCount)]
clients = np.ndarray(shape=(clientsCount, M), dtype=Client)
for i in range(clientsCount):
    for j in range(M):
        clients[i][j] = Client(epsilon)
WServer = WrappedServer(M, epsilon)
realF = np.zeros([changeRounds, M])
testMean = 0
numberMean = np.zeros(changeRounds * clientsCount)
selectedNumbers = np.zeros(changeRounds * clientsCount)

for i in range(changeRounds):
    for j in range(clientsCount):
        #Determine if we should change data or not:
        clientsValues[j] = np.random.randint(M)
        numberMean[i*clientsCount + j] = clientsValues[j]
        testMean += clientsValues[j]
        newValue = np.zeros(M)
        newValue[clientsValues[j]] = 1
        for k in range(M):
            toReport = newValue[k]
            toReport = int(toReport)
            # if toReport == 1:
            #     selectedNumbers[i*clientsCount + j] = clientsValues[j]
            # else:
            #     selectedNumbers[i*clientsCount + j] = 0
            [v, h] = clients[j][k].report(toReport)
            WServer.newValue(v, h, k)
            realF[i][k] += toReport
    # print("Sum of selected:", np.sum(selectedNumbers[i*clientsCount:i*clientsCount+clientsCount]))
    # print("Real sum:", np.sum(numberMean))
    # summationOfBits = 0
    # for index, val in enumerate(realF[i]):
    #     summationOfBits += val * index
    # print("Sum of server form:", summationOfBits);
    # print("Mean of selected: ", np.mean(selectedNumbers[i*clientsCount: i*clientsCount+clientsCount]));
    # testMean = realF[i].copy()
    # testMean /= (clientsCount/M)
    # summationOfBits = 0
    # for index2, number in enumerate(testMean):
    #     summationOfBits += (number*(clientsCount/M) * index2)
    # print("Sum of server form:", summationOfBits);
    # summationOfBits /= clientsCount
    # print( "Mean of server form:", summationOfBits)

    WServer.predicate()

realF /= (clientsCount)
result = WServer.finish()

# for index in range(changeRounds):  # calculating errors
#     error.append((result[index] - realF[index]) / realF[index] * 100)
#     print(index, "-> Estimated:", result[index], " Real:", realF[index], " Error: %", int(error[-1]))

realMean = 0
outputMean = 0
sumOfAllRoundsEstimations = 0

for index, row in enumerate(realF):
    for index2, number in enumerate(row):
        realMean += (number*(clientsCount) * index2)
realMean /= (clientsCount*changeRounds)

for index, row in enumerate(result):
    for index2, number in enumerate(row):
        outputMean += (number*(clientsCount) * index2)
outputMean /= (clientsCount*changeRounds)


for number in result[19]:
    sumOfAllRoundsEstimations += number


print('Consumed Differential Privacy Budget:', epsilon * changeRounds)
print("Global Mean difference:", abs(realMean - outputMean))
print("Result Mean:", outputMean)
print("Real Mean:", realMean)
print("Mean of numbers:", np.mean(numberMean))
print("Mean of sean values by server:", np.mean(selectedNumbers))
print("Sum of all estimated frequencies:", sumOfAllRoundsEstimations)

# print("Avg Error: %", np.mean(error))
for i in range(changeRounds):
    print(f"Min Squared Error at {i}'th iteration:", mean_squared_error(realF[i], result[i]))




