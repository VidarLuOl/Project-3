import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



np.random.seed(2018)
n = 50
maxdegree = 5
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(np.shape(x_train_scaled), np.shape(y_train))

for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    clf = model.fit(x_train_scaled,y_train)
    y_fit = clf.predict(x_train_scaled)
    y_pred = clf.predict(x_test_scaled) 
    polydegree[degree] = degree
    TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
    TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )

plt.plot(polydegree, TestError, label='Test Error')
plt.plot(polydegree, TrainError, label='Train Error')
plt.legend()
plt.show()



"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
"""





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler




def OLS(X_train, X_test, y_train, y_test, maxdegree):
    
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
    
    

def OLSnFeatures(X_train, X_test, y_train, y_test, maxdegree):
    m,n = np.shape(X_train)
    
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
    
    y_fit = 0
    y_pred = 0
    
    for degree in range(maxdegree):
        lin_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

        
        for feature in range(n):
            clf = lin_model.fit(x_train_scaled[:,feature].reshape(-1,1), y_train)
           
            y_fit += clf.predict(x_train_scaled[:,feature].reshape(-1,1))/n
            y_pred += clf.predict(x_test_scaled[:,feature].reshape(-1,1))/n
        
        polydegree[degree] = degree
        TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )
        
        error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
        variance[degree] = np.mean( np.var(y_pred) )


    plt.figure()
    plt.title("semilogy")
    plt.semilogy(polydegree, TestError, label='Test Error')
    plt.semilogy(polydegree, TrainError, label='Train Error')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("semilogy")
    plt.semilogy(polydegree, error, label='Error')
    plt.semilogy(polydegree, bias, label='Bias')
    plt.semilogy(polydegree, variance, label='Variance')
    plt.legend()
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



