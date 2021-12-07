import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Funksjoner.NN import *
from Funksjoner.LinearRegression import OLS
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

"""Variabler"""
H_layers = 2 # Antall hidden layers
H_nodes = 5 # Antall hidden nodes
"""_________"""

"""Lese inn data"""
Pandas_Bankruptcy = pd.read_csv("data.csv")
navn = Pandas_Bankruptcy.columns
y = np.array(Pandas_Bankruptcy)[:, 0] # (6819, )
x = np.array(Pandas_Bankruptcy)[:, 1:] # (6819, 95)
"""______________"""



"""
def NN():
    for i in range(0, len(true_input[0, :])):
       true_input[:, i] = true_input[:, i]/np.linalg.norm(true_input[:, i])

    train_X, test_X, train_Y, test_Y = train_test_split(true_output, true_input, test_size=0.7)

    

    Model = NeuralNetwork(train_X, train_Y, H_layers, H_nodes)
    Model.setUp()
"""


"""
if __name__ == "__main__":
    #NN()
    # BreastData()
    
    
    
"""

mse_train = []


for p in range(25):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    y_fit, y_pred, y_fit_scale, y_pred_scale, coefs, coefs_scale = OLS(x_train, x_test, y_train, y_test, p)
    
    mse_train.append(mean_squared_error(y_fit, y_train))


print(mse_train)
plt.plot(mse_train)

    
    
    
    
    