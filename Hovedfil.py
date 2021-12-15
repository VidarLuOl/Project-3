from matplotlib.pyplot import xlabel
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from Funksjoner.NN import * #Neural network
from Funksjoner.RV import * #Remove Variables
from Funksjoner.kNN import * #K nearest Neighbor
from Funksjoner.DT import * #Decision tree

"""Lese inn data"""
Pandas_Bankruptcy = pd.read_csv("data.csv")
navn = Pandas_Bankruptcy.columns
y = np.array(Pandas_Bankruptcy)[:, 0] # (6819, )
x = np.array(Pandas_Bankruptcy)[:, 1:] # (6819, 95)
np.random.seed(200)
"""______________"""


def NN():
    """Variabler"""
    h_layers = 3 # Antall hidden layers
    h_nodes = 5 # Antall hidden nodes
    batch = [100] # Størrelse på batch
    eta = [0.01]
    # Act = "SigmoidDer"
    """_________"""

    Model = Kern(Pandas_Bankruptcy, h_nodes, h_layers, batch, eta)
    Model.initializing()
    Model.compiler()
    Model.plots()

def kNN():
    """Variabler"""
    n_neighbors = 5
    loop = 8
    """_________"""

    Model = kNN_class(Pandas_Bankruptcy)
    Model.initializing()
    accuracyy =np.zeros(loop)
    r2 = np.zeros(loop)
    mse = np.zeros(loop)


    for i in range(0, loop):
        print("Number of Neighbors = ", i+n_neighbors)
        accuracyy[i], r2[i], mse[i]= Model.compile(i+n_neighbors)
        print()




    plt.plot(range(n_neighbors,n_neighbors+loop), accuracyy)
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(range(n_neighbors,n_neighbors+loop), r2)
    plt.xlabel("Number of Neighbors")
    plt.ylabel("R2_score")
    plt.show()
    plt.plot(range(n_neighbors,n_neighbors+loop), mse)
    plt.xlabel("Number of Neighbors")
    plt.ylabel("MSE")
    plt.show()

def dt():
    """Variabler"""
    """_________"""

    Model = Dec_tree(Pandas_Bankruptcy)
    Model.initializing()
    Model.compile()

def RV():
    """Variabler"""
    correlation_limit = 0.85 #Hvis correlasjonen mellom 2 features er over limit, så fjernes den ene
    PCA_n = 2 #Redusere dataen ned til n features
    p = 0.05 #P-verdien
    """_________"""

    Selection = RemoveVariable(Pandas_Bankruptcy)

    # Selection.Corre(correlation_limit)

    Selection.PCA(PCA_n)

    # Selection.BE(p)

    return Selection.data

if __name__ == "__main__":
    Pandas_Bankruptcy = RV() #Fjerner variabler, kan brukes samtidig som de andre funksjonene
    # print("New amount of features: ", np.shape(Pandas_Bankruptcy)[1], ", Shape: ", np.shape(Pandas_Bankruptcy))
    # NN() #Neural Network
    # kNN()
    # dt()