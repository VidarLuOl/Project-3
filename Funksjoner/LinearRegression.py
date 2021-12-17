import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import warnings
warnings.filterwarnings("ignore")


def OLS(X_train, X_test, y_train, y_test, maxdegree, nfeatures):
    """Linear regression
    
    This version of the OLS is able to make a model for a chosen feature, or a set of features
    It then plots the error for training and test data aswell as the R2,
    MSE, bias-variance and the accuracy
    """
    
    
    m,n = np.shape(X_train)

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
    for feature in nfeatures:
        for degree in range(maxdegree):
            lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    
            
            clf = lin_model.fit(x_train_scaled[:,feature].reshape(-1,1), y_train)
    
            y_fit = clf.predict(x_train_scaled[:,feature].reshape(-1,1))
            y_pred = clf.predict(x_test_scaled[:,feature].reshape(-1,1)) 
            
            polydegree[degree] = degree

            
            error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
            bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
            variance[degree] = np.mean(np.var(y_pred))
            
            rmse_train[degree] = (np.sqrt(mean_squared_error(y_train, y_fit)))
            r2_train[degree] = r2_score(y_train, y_fit)
            
            rmse_test[degree] = (np.sqrt(mean_squared_error(y_test, y_pred)))
            r2_test[degree] = r2_score(y_test, y_pred)
            
        
        
        plt.plot(polydegree, rmse_test, color = "blue")
        plt.plot(polydegree, rmse_train, color = "red")
    
    plt.title("RMS polynoms {}".format(nfeatures))
    plt.legend(["Test error", "Train error"])
    plt.xlabel("Polynom degree")
    plt.show()
    
    plt.figure()
    plt.title("R2")
    plt.plot(polydegree, r2_train, label='R2 train')
    plt.plot(polydegree, r2_test, label='R2 test')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Error, bias, variance")
    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label='Bias')
    plt.plot(polydegree, variance, label='Variance')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()

    return 0
    
    
    

def OLSnFeatures(X_train, X_test, y_train, y_test, maxdegree):
    """Linear regression
    
    This version of the OLS is able to make a model for all the features
    It is asumed that all of the features has the same influence
    It then plots the error for training and test data aswell as the R2,
    MSE, bias-variance and the accuracy
    """
    
    m,n = np.shape(X_train)

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

            
 
        for _ in y_pred:
            if _ <= 0:
                
 
                classification_error += 1

        score[degree] = classification_error/m
        
        error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[degree] = np.mean( np.var(y_pred) )
        
        rmse_train[degree] = (np.sqrt(mean_squared_error(y_train, y_fit)))
        r2_train[degree] = r2_score(y_train, y_fit)
        
        rmse_test[degree] = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_test[degree] = r2_score(y_test, y_pred)
    
    plt.figure()
    plt.title("RMS OLS")    
    plt.plot(polydegree, rmse_train, label='RMS train')
    plt.plot(polydegree, rmse_test, label='RMS test')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("RMS train OLS")    
    plt.plot(polydegree, rmse_train, label='RMS train')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS test OLS")    

    plt.plot(polydegree, rmse_test, label='RMS test')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()

    plt.figure()    
    plt.title("R2 OLS")
    plt.plot(polydegree, r2_train, label='R2 train')
    plt.plot(polydegree, r2_test, label='R2 test')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 OLS")
    plt.plot(polydegree, r2_train, label='R2 train')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 OLS")
    plt.plot(polydegree, r2_test, label='R2 test')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Error, bias, variance OLS")
    plt.semilogy(polydegree, error, label='Error')
    plt.semilogy(polydegree, bias, label='Bias')
    plt.semilogy(polydegree, variance, label='Variance')
    plt.xlabel("Poly degree")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Accuracy")
    plt.plot(polydegree, 1-score, label = 'score classification')
    plt.xlabel("Poly degree")
    plt.show()
    
    return 0
    
    
    
    
    
    
    
def RidgeRegression(X_train, X_test, y_train, y_test, input_lambdas, degree):
    """Ridge regression
    
    This version of the Ridge regression is able to make a model for all the features
    It is asumed that all of the features has the same influence
    It then plots the error for training and test data aswell as the R2,
    MSE, bias-variance and the accuracy  
    """
    
    m,n = np.shape(X_train)
    p = len(input_lambdas)
    
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
        
        error[i] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[i] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[i] = np.mean(np.var(y_pred) )
        
        rmse_train[i] = (np.sqrt(mean_squared_error(y_train, y_fit)))
        r2_train[i] = r2_score(y_train, y_fit)
        
        rmse_test[i] = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_test[i] = r2_score(y_test, y_pred)
        
  
    
    plt.figure()
    plt.title("RMS Ridge")    
    plt.loglog(lambdas, rmse_train, label='RMS train')
    plt.loglog(lambdas, rmse_test, label='RMS test')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS train Ridge")    
    plt.loglog(lambdas, rmse_train, label='RMS train')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("RMS test Ridge")    
    plt.loglog(lambdas, rmse_test, label='RMS test')
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 Ridge")
    plt.loglog(lambdas, r2_train, label='R2 train')
    plt.loglog(lambdas, r2_test, label='R2 test')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 train Ridge")
    plt.loglog(lambdas, r2_train, label='R2 train')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 test Ridge")
    plt.loglog(lambdas, r2_test, label='R2 test')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Error, bias, variance Ridge")
    plt.loglog(lambdas, error, label='Error')
    plt.loglog(lambdas, bias, label='Bias')
    plt.loglog(lambdas, variance, label='Variance')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Accuracy")
    plt.semilogx(lambdas, score, label = 'score classification')
    plt.xlabel("Lambda")
    plt.show()
    
    return 0

    
    
    
    
def LassoRegression(X_train, X_test, y_train, y_test, input_lambdas, degree):
    """Ridge regression
    
    This version of the Ridge regression is able to make a model for all the features
    It is asumed that all of the features has the same influence
    It then plots the error for training and test data aswell as the R2,
    MSE, bias-variance and the accuracy  
    """
    
    m,n = np.shape(X_train)
    p = len(input_lambdas)
    
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
        
        error[i] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[i] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[i] = np.mean(np.var(y_pred) )
        
        rmse_train[i] = (np.sqrt(mean_squared_error(y_train, y_fit)))
        r2_train[i] = r2_score(y_train, y_fit)
        
        rmse_test[i] = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_test[i] = r2_score(y_test, y_pred)
        
        
    plt.show()
    
    plt.figure()
    plt.title("RMS Lasso")    
    plt.semilogy(lambdas, rmse_train, label='RMS train')
    plt.semilogy(lambdas, rmse_test, label='RMS test')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()
    
    plt.figure()    
    plt.title("R2 Lasso")
    plt.semilogy(lambdas, r2_train, label='R2 train')
    plt.semilogy(lambdas, r2_test, label='R2 test')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Error, bias, variance Lasso")
    plt.loglog(lambdas, error, label='Error')
    plt.loglog(lambdas, bias, label='Bias')
    plt.loglog(lambdas, variance, label='Variance')
    plt.xlabel("Lambda")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Accuracy")
    plt.semilogx(lambdas, score, label = 'score classification')
    plt.xlabel("Lambda")
    plt.show()
    
    return 0
    
    
    
    
    
    
    
    
    
    
    
    