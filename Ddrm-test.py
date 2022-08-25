import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error
import mmh3
from WrappedServer import WrappedServer

epsilon = 1
clientsCount = 10000
changeRounds = 10
numberOfChangesDebug = 0
sparcity = 0.75
error = []
Data = ['contrary', 'popular', 'belief', 'lorem', 'ipsum', 'simply', 'random', 'text', 'latin', 'literature']
hashedSize = 12
hashedValues = dict()
numberOfHashFunctionsInBloomFilter = 4
poolBloomFilter = []
for i in Data:
    hashedValue = np.zeros(hashedSize)
    for j in range(numberOfHashFunctionsInBloomFilter):
        hashedValue[mmh3.hash(i, j) % hashedSize] = 1    
    hashedValues[i] = hashedValue
    poolBloomFilter.append(hashedValue)
print(hashedValues)
clientsValues = np.random.choice(Data, size=(clientsCount))
clients = [Client(epsilon) for i in range(clientsCount)]
WServer = WrappedServer(hashedSize, epsilon);
realF = np.zeros([changeRounds, hashedSize])

for i in range(changeRounds):
    for j in range(clientsCount):
        #Determine if we should change data or not:
        valueChangedP = np.random.rand()
        if valueChangedP >= sparcity:
            numberOfChangesDebug += 1
            clientsValues[j] = np.random.choice(Data)
        [v, h] = clients[j].report(hashedValues[clientsValues[j]][j % hashedSize])
        WServer.newValue(v, h, j%hashedSize)
        realF[i][j % hashedSize] += hashedValues[clientsValues[j]][j % hashedSize]
    WServer.predicate()

realF /= (clientsCount/hashedSize)
result = WServer.finish()


print(realF)
print(result)
print(numberOfChangesDebug)

# for index in range(changeRounds):  # calculating errors
#     error.append((result[index] - realF[index]) / realF[index] * 100)
#     print(index, "-> Estimated:", result[index], " Real:", realF[index], " Error: %", int(error[-1]))

# print("Avg Error: %", np.mean(error))
for i in range(changeRounds):
    print(f"Min Squared Error at {i}'th iteration:", mean_squared_error(realF[i], result[i]))




