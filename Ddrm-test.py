import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error


epsilon = 1
clientsCount = 10000
changeRounds = 10
numberOfChangesDebug = 0
sparcity = 0.75
error = []
clientsValues = np.random.randint(2, size=(clientsCount))
clients = [Client(epsilon) for i in range(clientsCount)]
server = Server(epsilon)
realF = np.zeros(changeRounds)

for i in range(changeRounds):
    for j in range(clientsCount):
        #Determine if we should change data or not:
        valueChangedP = np.random.rand()
        if valueChangedP >= sparcity:
            numberOfChangesDebug += 1
            clientsValues[j] = (clientsValues[j] + 1) % 2
        [v, h] = clients[j].report(clientsValues[j])
        server.newValue(v, h)
        realF[i] += clientsValues[j]
    server.predicate()

realF /= clientsCount
result = server.finish()
#Remove dummy data.
result.reverse()
result.pop()
result.reverse()

print(realF)
print(result)
print(numberOfChangesDebug)

for index in range(changeRounds):  # calculating errors
    error.append((result[index] - realF[index]) / realF[index] * 100)
    print(index, "-> Estimated:", result[index], " Real:", realF[index], " Error: %", int(error[-1]))

print("Avg Error: %", np.mean(error))
print("Min Squared Error:", mean_squared_error(realF, result))




