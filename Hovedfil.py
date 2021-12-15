import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Funksjoner.NN import *
from Funksjoner.LinearRegression import OLS, OLSnFeatures, RidgeRegression, LassoRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
"""
import warnings
warnings.filterwarnings("ignore")
"""
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


"""
RMSE_train = []
RMSE_test = []
R2_train = []
R2_test = []
Y_train_pred = []
Y_test_pred = []
poly = []

for p in range(10):
    np.random.seed(2019)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    y_train_predict, y_test_predict, rmse_train, r2_train, rmse_test, r2_test = OLS(x_train, x_test, y_train, y_test, p)
    #y_train_predict, y_test_predict = OLS(x_train, x_test, y_train, y_test, p)

    RMSE_train.append(rmse_train)
    RMSE_test.append(rmse_test)
    R2_train.append(r2_train)
    R2_test.append(r2_test)


    Y_train_pred.append(y_train_predict)
    Y_test_pred.append(y_test_predict)
    poly.append(p)
    

plt.figure()
plt.plot(poly, RMSE_train)
plt.plot(poly, RMSE_test)
plt.show()

plt.figure()
plt.plot(poly, R2_train)
plt.plot(poly, R2_test)
plt.show()


plt.figure()
plt.plot(poly, Y_train_pred)
plt.plot(poly, Y_test_pred)
plt.show()

"""


np.random.seed(2019)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
OLS(x_train, x_test, y_train, y_test, 20, 3)


"""
np.random.seed(2019)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
OLSnFeatures(x_train, x_test, y_train, y_test, 26)
"""

"""
#good_rgn = [742873, 515295, 837215, 1763]
np.random.seed(2019)
lambdas = np.logspace(-15, 1.5, 40)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
RidgeRegression(x_train, x_test, y_train, y_test, lambdas, 10)


np.random.seed(2019)
lambdas = np.logspace(-15, 1.5, 40)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
LassoRegression(x_train, x_test, y_train, y_test, lambdas, 10)
"""
    
    
    
    
    
    
    
    
    
    
    
    