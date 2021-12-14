from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error


class kNN_class:
    #https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
    def __init__(self, datafile):
        self.data = datafile
    def initializing(self):
        SC = StandardScaler()
        #Deler inn i set med resultater og set med variabler
        self.X = self.data.iloc[:, 1:]
        self.y = self.data.iloc[:, 0]

        #Skalerer dataen
        self.X = SC.fit_transform(self.X)

    def compile(self, nn):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)

        knn = KNeighborsClassifier(n_neighbors = nn)
        knn.fit(X_train,y_train)

        y_pred = knn.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        # print(cm)

        acc = accuracy_score(y_test, y_pred)
        # print(acc)

        r2 = r2_score(y_test, y_pred)
        # print(r2)

        mse = mean_squared_error(y_test, y_pred)
        # print(mse) 


        return acc, r2, mse
