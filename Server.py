import math
import numpy as np 


class Server:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.coef = (1 + math.exp(self.epsilon))/(math.exp(self.epsilon) - 1)
        self.sumVOf1 = 0
        self.sumOfUsersOf1 = 0
        self.sumVOfh = 0
        self.sumOfUsersOfh = 0
        self.f1 = 0
        self.f2 = 0
        self.f = [0]
        self.varianceF = [0]
        self.t = 0
        self.lastRoot = 0
    
    def newValue(self, v, h):
        callibratedV = v * self.coef
        self.lastRoot = max(self.lastRoot, h)
        if h == 0:
            self.sumOfUsersOf1 += 1
            self.sumVOf1 += callibratedV
        else:
            self.sumOfUsersOfh += 1
            self.sumVOfh += callibratedV

    def varianceF1(self):
        varF1 = self.varianceF[len(self.varianceF) - 1] + \
                (((math.exp(self.epsilon) + 1)/(math.exp(self.epsilon) - 1))**2) / \
                    self.sumOfUsersOf1
        return varF1

    def varianceF2(self):
        tPrime = self.t - 2**self.lastRoot
        varF2 = self.varianceF[tPrime] + \
                (((math.exp(self.epsilon) + 1)/(math.exp(self.epsilon) - 1)) ** 2) / \
                    self.sumOfUsersOfh
        return varF2
        
    def computeVariance(self):
        if self.t % 2 == 0:
            vf1 = self.varianceF1()
            vf2 = self.varianceF2()
            return (vf1 * vf2)/(vf1 + vf2)
        else:
            return self.varianceF[len(self.varianceF) - 1] + \
                    (((math.exp(self.epsilon) + 1)/(math.exp(self.epsilon) - 1)) ** 2) / \
                        self.sumOfUsersOf1
            
    def computeW1(self):
        varf1 = self.varianceF1()
        return math.pow(varf1, -1)
    def computeW2(self):
        varf2 = self.varianceF2()
        return math.pow(varf2, -1)
    def computeW(self):
        w1 = self.computeW1()
        w2 = self.computeW2()
        return w1/(w1 + w2)
    def predicate(self):
        self.t += 1
        if self.t % 2 == 0:
            self.f1 = self.f[len(self.f) - 1] + \
                        (self.sumVOf1 / self.sumOfUsersOf1)
            tPrime = self.t - 2**self.lastRoot
            self.f2 = self.f[tPrime] + \
                        (self.sumVOfh / self.sumOfUsersOfh)
        else:
            self.f1 = self.f2 = self.f[len(self.f) - 1] + \
                        (self.sumVOf1 / self.sumOfUsersOf1)
        if self.t % 2 == 0:
            w = self.computeW()
        else:
            w = 0.5 #Just to neutralize its effect.
        self.f.append(w * self.f1 + (1 - w) * self.f2)
        varF = self.computeVariance()
        self.varianceF.append(varF)
        #Reset state of server.
        self.sumVOf1 = 0
        self.sumOfUsersOf1 = 0
        self.sumVOfh = 0
        self.sumOfUsersOfh = 0
    def finish(self):
        return self.f