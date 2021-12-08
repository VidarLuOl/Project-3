import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def OLS(X_train, X_test, y_train, y_test, degree):
    
    m,n = np.shape(X_train)
    Y_train = np.zeros((m,n))
    
    #print(m,n, np.shape(y_train))
    for j in range(n):
        for i in range(m):
            Y_train[i,j] = y_train[i]
            

    
    # model evaluation for training set
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #print(np.shape(X_train_scaled), np.shape(y_train.reshape(-1,1)))
    
    lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    #clf = lin_model.fit(X_train_scaled[:,1].reshape(-1,1), y_train)
    
    clf = lin_model.fit(X_train_scaled.reshape(-1,1), Y_train.ravel())
    
    
    y_train_predict = clf.predict(X_train_scaled.reshape(-1,1))
    #rmse_train = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    #r2_train = r2_score(y_train, y_train_predict)
    
    """
    print("The model performance for training set polynom:", degree)
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse_train))
    print('R2 score is {}'.format(r2_train))
    print("\n")
    """


    y_test_predict = lin_model.predict(X_test_scaled[:,0].reshape(-1,1))
    #rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    #r2_test = r2_score(y_test, y_test_predict)
    
    """
    print("The model performance for testing set polynom:", degree)
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse_test))
    print('R2 score is {}'.format(r2_test))
    """
    
    #plt.scatter(Y_test, y_test_predict)

    

    return y_train_predict, y_test_predict#, rmse_train, r2_train, rmse_test, r2_test

def OLSpolynomial(X_train, X_test, y_train, y_test, maxdegree):
    
    TestError = np.zeros(maxdegree)
    TrainError = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    error = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
        
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    
    for degree in range(maxdegree):
        lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

        
        clf = lin_model.fit(x_train_scaled[:,0].reshape(-1,1), y_train)
        
        y_fit = clf.predict(x_train_scaled[:,0].reshape(-1,1))
        y_pred = clf.predict(x_test_scaled[:,0].reshape(-1,1)) 
        
        polydegree[degree] = degree
        TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )
        
        error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[degree] = np.mean( np.var(y_pred) )
        
    plt.figure()
    plt.plot(polydegree, TestError, label='Test Error')
    plt.plot(polydegree, TrainError, label='Train Error')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label='Bias')
    plt.plot(polydegree, variance, label='Variance')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    