import numpy as np
from Client import Client

class WrappeedClient:
    def __init__(self, M, epsilon):
        self.M = M
        self.epsilon = epsilon
        self.Clients = [Client(epsilon) for i in range(M)]
        self.changes = 0
        self.prevValue = -1
    
    def report(self, value):
        if(value != self.prevValue):
            self.changes+=1
        self.prevValue = value
        binaryRepresentation = f'{value:0{self.M}b}'
        characterized = [c for c in binaryRepresentation]
        allV = []
        allH = []
        for i in range(self.M):
            toReport = int(characterized[i])
            [v, h] = self.Clients[i].report(toReport)
            allV.append(v)
            allH.append(h)
        return [allV, allH]
    def howManyChanges(self):
        return self.changes