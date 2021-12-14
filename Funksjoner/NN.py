import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# from keras import Sequential
# from keras.layers import Dense
# from keras import optimizers

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function


class Kern:
    #https://medium.datadriveninvestor.com/building-neural-network-using-keras-for-classification-3a3656c726c1
    def __init__(self, datafile, h_nodes, h_layers, batchsize, eta):
        self.data = datafile
        self.h_nodes = h_nodes
        self.h_layers = h_layers
        self.batchsize = batchsize
        self.eta = eta

    def plots(self):
        # self.data.drop(self.data.columns[7:96], axis=1, inplace=True)
        # print(self.data)
        # print(self.data.columns[5])
        # print(self.data.describe(include="all"))
        # sns.pairplot(self.data, hue="Bankrupt?")
        # sns.heatmap(self.data.corr(), annot=True)
        # plt.show()
        sns.countplot("Bankrupt?" , data=self.data)
        plt.plot()
        # print(self.data)

    def initializing(self):
        SC = StandardScaler()
        #Deler inn i set med resultater og set med variabler
        self.X = self.data.iloc[:, 1:]
        self.y = self.data.iloc[:, 0]

        #Skalerer dataen
        self.X = SC.fit_transform(self.X)

        #Lager Neuralt Network modellen
        self.layers = Sequential()
        for i in range(self.h_layers):
            self.layers.add(Dense(self.h_nodes, activation="relu", kernel_initializer="random_normal"))
        self.layers.add(Dense(1, activation="relu", kernel_initializer="random_normal"))

    def compiler(self):
        #Spiltter dataen
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)

        #Lager matrise til å innholde verdier til hver nye batch og eta
        NN_keras = np.zeros((len(self.eta), len(self.batchsize)), dtype=object)
        # print(self.eta)
        # print(1-np.sum(self.y)/len(self.y))
        # Compiles dataen, altså spesifiser optimiseringen, loss funksjonen og metrics
        for i, eta_tmp in enumerate(self.eta):
            sgd = optimizers.SGD(learning_rate=eta_tmp)
            self.layers.compile(optimizer = sgd, loss="binary_crossentropy", metrics=["accuracy"])
            for j, batchsize_tmp in enumerate(self.batchsize):
                NN_keras[i][j] = self.layers.fit(X_train, y_train, batch_size=batchsize_tmp, epochs=10, shuffle=True, verbose=0)
                self.layers.evaluate(X_test, y_test)

                y_pred = self.layers.predict(X_test)
                print(confusion_matrix(y_test, y_pred))


        # y_pred = self.layers.predict(X_test)
        # y_pred = (y_pred > 0.5)

        # cm = confusion_matrix(y_test, y_pred)
        
        
        














class NeuralNetwork():
    def __init__(self, X, Y, h_layers, h_nodes, batch):
        self.X = X
        self.h_layers = h_layers
        self.h_nodes = h_nodes
        self.loss = np.empty(batch)
        self.batch = batch


    def setUp(self):
        self.layers = [] #np.empty for kjappere programering?
        
        #Legger til andre layer, ikke første fordi første er input og kan dermed ikke forandres
        self.layers.append(layer(np.shape(self.X)[1], self.h_nodes))

        #Legger til layersene mellom andre og siste
        for i in range(self.h_layers-1):
            self.layers.append(layer(self.h_nodes, self.h_nodes))

        #Legger til siste layer
        self.layers.append(layer(self.h_nodes, 2))


    def FF(self, X, Act): #Feed Forward
        #Fast forward, finner ny verdi i første hidden layer
        self.layers[0].z = (self.layers[0].weights@X + self.layers[0].bias.T).T
        self.layers[0].a = getattr(Activation, Act)(self.layers[0].z)

        #Fast forward, finner ny verdi i resten av hidden layers
        for i in range(1, self.h_layers):
            self.layers[i].z = self.layers[i].weights@self.layers[i-1].a# + self.layers[i].bias
            self.layers[i].a = getattr(Activation, Act)(self.layers[i].z)


        #Fast forward, finner ny verdi i output
        self.layers[-1].z = (self.layers[-1].weights@self.layers[-2].delta + self.layers[-1].bias)
        self.layers[-1].a = getattr(Activation, Act)(self.layers[-1].z)





    def BP(self, Y): #Backward Propagation
        #Cost er feilen for hele treningssettet, LOSS er for et eksempel
        self.loss = Loss_Func.BCE(self.layers[-1].a.T, Y)*Activation.SigmoidDer(self.layers[-1].z)

        # self.layers[-1].delta = (self.layers[-1].a - Y) @ Activation.
        print(self.loss)
        print(Y)
        print(self.layers[-1].a.T)
        print(Y -  self.layers[-1].a.T)
        # print(tf.keras.losses.binary_crossentropy(self.layers[-1].a), Y, logits=True).numpy()
        
class layer:
    def __init__(self, nodesfrom, nodesto):
        self.delta = np.random.randn(nodesto,1) #Må være 1 for å kunne transponere
        self.weights = np.random.randn(nodesto, nodesfrom)
        self.bias = np.random.randn(nodesto,1) #-||-
        self.z = np.random.randn(nodesto,1)
        self.a = np.random.randn(nodesto,1)

class Activation:
    def Sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def SigmoidDer(z):
        return np.exp(-z)/(1+np.exp(-z))**2

    def Identity(z):
        return z

    def IdentityDer(z):
        return np.ones(z.shape)

class Loss_Func:
    def BCE(ytilde, y):
        #BCE https://medium.com/artificialis/neural-network-basics-loss-and-cost-functions-9d089e9de5f8
        return (-1/len(ytilde))*np.sum(y * np.log(ytilde) + (1 - y) * np.log(1 - ytilde))