import numpy as np

class Activation:
    def Sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def SigmoidDer(z):
        return np.exp(-z)/(1+np.exp(-z))**2