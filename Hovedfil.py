import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Funksjoner.NN import *

"""Variabler"""
H_layers = 2 # Antall hidden layers
H_nodes = 5 # Antall hidden nodes
"""_________"""

"""Lese inn data"""
Pandas_Bankruptcy = pd.read_csv("data.csv")
navn = Pandas_Bankruptcy.columns
true_output = np.array(Pandas_Bankruptcy)[:, 0] # (6819, 95)
true_input = np.array(Pandas_Bankruptcy)[:, 1:] # (6819, )
"""______________"""


def NN():
    for i in range(0, len(true_input[0, :])):
       true_input[:, i] = true_input[:, i]/np.linalg.norm(true_input[:, i])

    train_X, test_X, train_Y, test_Y = train_test_split(true_output, true_input, test_size=0.7)

    

    Model = NeuralNetwork(train_X, train_Y, H_layers, H_nodes)
    Model.setUp()



if __name__ == "__main__":
    NN()
    # BreastData()