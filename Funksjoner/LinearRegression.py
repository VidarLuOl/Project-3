import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
"""
import warnings
warnings.filterwarnings("ignore")
"""

def OLSstoopid(X_train, X_test, y_train, y_test, degree):
    
    m,n = np.shape(X_train)

    
    # model evaluation for training set
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

    clf = lin_model.fit(X_train_scaled[:,0].reshape(-1,1), y_train)
    

    y_train_predict = clf.predict(X_train_scaled[:,0].reshape(-1,1))
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    r2_train = r2_score(y_train, y_train_predict)
    
    """
    print("The model performance for training set polynom:", degree)
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse_train))
    print('R2 score is {}'.format(r2_train))
    print("\n")
    """


    y_test_predict = lin_model.predict(X_test_scaled[:,0].reshape(-1,1))
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2_test = r2_score(y_test, y_test_predict)
    
    """
    print("The model performance for testing set polynom:", degree)
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse_test))
    print('R2 score is {}'.format(r2_test))
    """
    
    #plt.scatter(Y_test, y_test_predict)

    

    return y_train_predict, y_test_predict, rmse_train, r2_train, rmse_test, r2_test



def OLS(X_train, X_test, y_train, y_test, maxdegree, nfeatures):
    m,n = np.shape(X_train)
    TestError = np.zeros(maxdegree)
    TrainError = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    error = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    rmse_train = np.zeros(maxdegree)
    rmse_test = np.zeros(maxdegree)
    r2_train = np.zeros(maxdegree)
    r2_test = np.zeros(maxdegree)
      
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    
    
    plt.figure()
    for feature in range(nfeatures):
        for degree in range(maxdegree):
            lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    
            
            clf = lin_model.fit(x_train_scaled[:,feature].reshape(-1,1), y_train)
    
            y_fit = clf.predict(x_train_scaled[:,feature].reshape(-1,1))
            y_pred = clf.predict(x_test_scaled[:,feature].reshape(-1,1)) 
            
            polydegree[degree] = degree
            TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
            TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )
            
            error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
            bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
            variance[degree] = np.mean(np.var(y_pred))
            
            rmse_train[degree] = (np.sqrt(mean_squared_error(y_train, y_fit)))
            r2_train[degree] = r2_score(y_train, y_fit)
            
            rmse_test[degree] = (np.sqrt(mean_squared_error(y_test, y_pred)))
            r2_test[degree] = r2_score(y_test, y_pred)
            
        
        plt.title("Error")
        plt.semilogy(polydegree, TestError, color = "blue")
        plt.semilogy(polydegree, TrainError, color = "red")

    plt.show()
        
    """
    plt.figure()
    plt.title("Error")
    plt.semilogy(polydegree, TestError, label='Test Error')
    plt.semilogy(polydegree, TrainError, label='Train Error')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS")    
    plt.plot(polydegree, rmse_train, label='RMS train')
    plt.plot(polydegree, rmse_test, label='RMS test')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("R2")
    plt.plot(polydegree, r2_train, label='R2 train')
    plt.plot(polydegree, r2_test, label='R2 test')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Error, bias, variance")
    plt.loglog(polydegree, error, label='Error')
    plt.loglog(polydegree, bias, label='Bias')
    plt.loglog(polydegree, variance, label='Variance')
    plt.legend()
    plt.show()
    """ 
    return 0
    
    
    

def OLSnFeatures(X_train, X_test, y_train, y_test, maxdegree):
    m,n = np.shape(X_train)
    
    TestError = np.zeros(maxdegree-1)
    TrainError = np.zeros(maxdegree-1)
    polydegree = np.linspace(1, maxdegree-1, maxdegree-1)
    error = np.zeros(maxdegree-1)
    variance = np.zeros(maxdegree-1)
    bias = np.zeros(maxdegree-1)
    score = np.zeros(maxdegree-1)

    rmse_train = np.zeros(maxdegree-1)
    rmse_test = np.zeros(maxdegree-1)
    r2_train = np.zeros(maxdegree-1)
    r2_test = np.zeros(maxdegree-1)
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    


    for degree in range(1,maxdegree-1):
        y_fit = 0
        y_pred = 0
        classification_error = 0
        lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

        for feature in range(n):
            clf = lin_model.fit(x_train_scaled[:,feature].reshape(-1,1), y_train)
           
            y_fit += clf.predict(x_train_scaled[:,feature].reshape(-1,1))/n
            y_pred += clf.predict(x_test_scaled[:,feature].reshape(-1,1))/n

            
        #print("")
        for _ in y_pred:
            if _ <= 0:
                
                #print(_)
                classification_error += 1
        #print("")
        score[degree] = classification_error/m

        TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )
        
        error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[degree] = np.mean( np.var(y_pred) )
        
        rmse_train[degree] = (np.sqrt(mean_squared_error(y_train, y_fit)))
        r2_train[degree] = r2_score(y_train, y_fit)
        
        rmse_test[degree] = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_test[degree] = r2_score(y_test, y_pred)


    """
    plt.figure()
    plt.title("Error OLS")
    plt.semilogy(polydegree, TestError, label='Test Error')
    plt.semilogy(polydegree, TrainError, label='Train Error')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS OLS")    
    plt.plot(polydegree, rmse_train, label='RMS train')
    plt.plot(polydegree, rmse_test, label='RMS test')
    plt.legend()
    plt.show()

    plt.figure()    
    plt.title("R2 OLS")
    plt.plot(polydegree, r2_train, label='R2 train')
    plt.plot(polydegree, r2_test, label='R2 test')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Error, bias, variance OLS")
    plt.loglog(polydegree, error, label='Error')
    plt.loglog(polydegree, bias, label='Bias')
    plt.loglog(polydegree, variance, label='Variance')
    plt.legend()
    plt.show()
    """
    plt.figure()
    plt.plot(polydegree, 1-score, label = 'score classification')
    plt.show()
    
    return 0
    
    
    
    
    
    
    
def RidgeRegression(X_train, X_test, y_train, y_test, input_lambdas, degree):
    m,n = np.shape(X_train)
    p = len(input_lambdas)
    
    TestError = np.zeros(p)
    TrainError = np.zeros(p)
    lambdas = np.zeros(p)
    error = np.zeros(p)
    variance = np.zeros(p)
    bias = np.zeros(p)
    rmse_train = np.zeros(p)
    rmse_test = np.zeros(p)
    r2_train = np.zeros(p)
    r2_test = np.zeros(p)
    score = np.zeros(p)
     
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)   
    

    
    for i in range(p):
        lamb = input_lambdas[i]
        lin_model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=lamb, fit_intercept=False))
        y_fit = 0
        y_pred = 0
        classification_error = 0
        
        for feature in range(n):
            clf = lin_model.fit(x_train_scaled[:,feature].reshape(-1,1), y_train)
           
            y_fit += clf.predict(x_train_scaled[:,feature].reshape(-1,1))/n
            y_pred += clf.predict(x_test_scaled[:,feature].reshape(-1,1))/n

        for _ in y_pred:
            if _ <= 0:
                classification_error += 1
                
        score[i] = classification_error/m
        
        lambdas[i] = lamb
        TestError[i] = np.mean( np.mean((y_test - y_pred)**2) )
        TrainError[i] = np.mean( np.mean((y_train - y_fit)**2) )
        
        error[i] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[i] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[i] = np.mean(np.var(y_pred) )
        
        rmse_train[i] = (np.sqrt(mean_squared_error(y_train, y_fit)))
        r2_train[i] = r2_score(y_train, y_fit)
        
        rmse_test[i] = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_test[i] = r2_score(y_test, y_pred)
        
    """
    plt.figure()
    plt.title("Error Ridge")
    plt.semilogy(lambdas, TestError, label='Test Error')
    plt.semilogy(lambdas, TrainError, label='Train Error')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS Ridge")    
    plt.semilogy(lambdas, rmse_train, label='RMS train')
    plt.semilogy(lambdas, rmse_test, label='RMS test')
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 Ridge")
    plt.semilogy(lambdas, r2_train, label='R2 train')
    plt.semilogy(lambdas, r2_test, label='R2 test')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Error, bias, variance Ridge")
    plt.loglog(lambdas, error, label='Error')
    plt.loglog(lambdas, bias, label='Bias')
    plt.loglog(lambdas, variance, label='Variance')
    plt.legend()
    plt.show()
    """
    plt.figure()
    plt.semilogx(lambdas, score, label = 'score classification')
    plt.show()
    
    return 0

    
    
    
    
def LassoRegression(X_train, X_test, y_train, y_test, input_lambdas, degree):
    m,n = np.shape(X_train)
    p = len(input_lambdas)
    
    TestError = np.zeros(p)
    TrainError = np.zeros(p)
    lambdas = np.zeros(p)
    error = np.zeros(p)
    variance = np.zeros(p)
    bias = np.zeros(p)
    rmse_train = np.zeros(p)
    rmse_test = np.zeros(p)
    r2_train = np.zeros(p)
    r2_test = np.zeros(p)
    score = np.zeros(p)
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)   
    

    
    for i in range(p):
        lamb = input_lambdas[i]
        lin_model = make_pipeline(PolynomialFeatures(degree=degree), Lasso(alpha=lamb, fit_intercept=False))
        y_fit = 0
        y_pred = 0
        classification_error = 0
        
        for feature in range(n):
            clf = lin_model.fit(x_train_scaled[:,feature].reshape(-1,1), y_train)
           

            y_fit += clf.predict(x_train_scaled[:,feature].reshape(-1,1))/n
            y_pred += clf.predict(x_test_scaled[:,feature].reshape(-1,1))/n

        for _ in y_pred:
            if _ <= 0:
                
                classification_error += 1
                
        score[i] = classification_error/m
        
        lambdas[i] = lamb
        TestError[i] = np.mean( np.mean((y_test - y_pred)**2) )
        TrainError[i] = np.mean( np.mean((y_train - y_fit)**2) )
        
        error[i] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[i] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[i] = np.mean(np.var(y_pred) )
        
        rmse_train[i] = (np.sqrt(mean_squared_error(y_train, y_fit)))
        r2_train[i] = r2_score(y_train, y_fit)
        
        rmse_test[i] = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_test[i] = r2_score(y_test, y_pred)
        
        
    """
    plt.figure()
    plt.title("Error Lasso")
    plt.semilogy(lambdas, TestError, label='Test Error')
    plt.semilogy(lambdas, TrainError, label='Train Error')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS Lasso")    
    plt.semilogy(lambdas, rmse_train, label='RMS train')
    plt.semilogy(lambdas, rmse_test, label='RMS test')
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 Lasso")
    plt.semilogy(lambdas, r2_train, label='R2 train')
    plt.semilogy(lambdas, r2_test, label='R2 test')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Error, bias, variance Lasso")
    plt.loglog(lambdas, error, label='Error')
    plt.loglog(lambdas, bias, label='Bias')
    plt.loglog(lambdas, variance, label='Variance')
    plt.legend()
    plt.show()
    """
    plt.figure()
    plt.semilogx(lambdas, score, label = 'score classification')
    plt.show()
    
    return 0
    
    
    
    
    
    
    
    
    
    
    
    