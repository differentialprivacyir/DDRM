import numpy as np
from Client import Client

class WrappeedClient:
    def __init__(self, M, epsilon):
        self.M = M
        self.epsilon = epsilon
        self.Clients = [Client(epsilon) for i in range(M)]
        self.changes = 0
    
    def report(self, value):
        budgetConsumed = False
        binaryRepresentation = f'{value:0{self.M}b}'
        characterized = [c for c in binaryRepresentation]
        allV = []
        allH = []
        for i in range(self.M):
            toReport = int(characterized[i])
            [v, h] = self.Clients[i].report(toReport)
            budgetConsumed = budgetConsumed or self.Clients[i].budgetConsumptionInLastReport()
            allV.append(v)
            allH.append(h)
        if budgetConsumed:
            self.changes += 1
        return [allV, allH]
    def howManyChanges(self):
        return self.changes