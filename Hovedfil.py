from matplotlib.pyplot import xlabel
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from Funksjoner.NN import * #Neural network
from Funksjoner.RV import * #Remove Variables
from Funksjoner.kNN import * #K nearest Neighbor
from Funksjoner.DT import * #Decision tree
from Funksjoner.LinearRegression import OLS, OLSnFeatures, RidgeRegression, LassoRegression #Linear Regression
import warnings
warnings.filterwarnings("ignore")

"""Lese inn data"""
Pandas_Bankruptcy = pd.read_csv("data.csv")
navn = Pandas_Bankruptcy.columns
y = np.array(Pandas_Bankruptcy)[:, 0] # (6819, )
x = np.array(Pandas_Bankruptcy)[:, 1:] # (6819, 95)
np.random.seed(200)
"""______________"""

def LR():
    """____OLS feature(s)_____"""

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2019)
    OLS(x_train, x_test, y_train, y_test, 20, [0,1,2])

    """_____OLS all features_____"""


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=2019)
    OLSnFeatures(x_train, x_test, y_train, y_test, 26)


    """____Ridge regression____"""

    lambdas = np.logspace(-15, 1.5, 50)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=2019)
    RidgeRegression(x_train, x_test, y_train, y_test, lambdas, 10)


    """____Lasso regression____"""

    lambdas = np.logspace(-15, 1.5, 40)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=2019)
    LassoRegression(x_train, x_test, y_train, y_test, lambdas, 10)

def NN():
    """Variabler"""
    h_layers = 7 # Antall hidden layers
    h_nodes = 12 # Antall hidden nodes
    batch = [10, 50, 100, 500] # Størrelse på batch
    eta = [0.1, 0.01, 0.001]
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

    # Selection.PCA(PCA_n)

    Selection.BE(p)

    return Selection.data

if __name__ == "__main__":
    Pandas_Bankruptcy = RV() #Fjerner variabler, kan brukes samtidig som de andre funksjonene
    # print("New amount of features: ", np.shape(Pandas_Bankruptcy)[1], ", Shape: ", np.shape(Pandas_Bankruptcy))
    LR() #Linear Regression
    # NN() #Neural Network
    # kNN()
    # dt()