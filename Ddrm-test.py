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
M = 12
clientsValues = np.random.randint(2**M, size=clientsCount)
clients = [Client(epsilon) for i in range(clientsCount)]
WServer = WrappedServer(M, epsilon);
realF = np.zeros([changeRounds, M])

for i in range(changeRounds):
    for j in range(clientsCount):
        #Determine if we should change data or not:
        valueChangedP = np.random.rand()
        if valueChangedP >= sparcity:
            numberOfChangesDebug += 1
            clientsValues[j] = np.random.randint(2**M)
        toReport = f'{clientsValues[j]:0{M}b}'[j % M]
        toReport = int(toReport)
        [v, h] = clients[j].report(toReport)
        WServer.newValue(v, h, j%M)
        realF[i][j % M] += toReport
    WServer.predicate()

realF /= (clientsCount/M)
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




