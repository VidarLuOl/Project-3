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
    
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    


            
            
    print(np.shape(X_train), np.shape(y_train))
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

    clf = model.fit(X_train_scaled, y_train)

    z_fit = clf.predict(X_train_scaled)
    z_pred = clf.predict(X_test_scaled)


    return z_fit, z_pred, clf.coef_

