from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split



class RemoveVariable:
    def __init__(self, data):
        self.data = data


    # def start(self):
    #     points = np.shape(self.X)[0]
    #     h = 95/points
    #     x_min, x_max = 0,1
    #     y_min, y_max = 0,1

    #     data = datasets.load_iris()

    #     print(np.shape(data.data))
    #     print((9/0.01))
    #     print(np.shape(self.X))
    #     X_pca = PCA().fit_transform(self.X)
    #     X_selected = X_pca[:, :2]

    #     clf = SVC(kernel="linear")
    #     clf.fit(X_selected, self.Y)

    #     cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    #     cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    #     plt.figure()
    #     plt.title('PCA - Iris dataset')
    #     plt.xlabel('Dimension 1')
    #     plt.ylabel('Dimension 2')
    #     plt.scatter(X_pca[:,0],X_pca[:,1],c=self.Y,cmap=cmap_bold)
    #     plt.show()

    def Corre(self, correlation_limit):
        #https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
        SC = StandardScaler()
        X = self.data.drop("Bankrupt?", axis=1)
        y = self.data["Bankrupt?"]

        # print(X)

        #Skalerer dataen
        X_scaled = pd.DataFrame(SC.fit_transform(X))

        X = X_scaled.rename(columns={i:j for i,j in zip(X_scaled.columns,X.columns)})

        cor_matrix = X.corr().abs()

        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        
        # print(upper_tri)

        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_limit)]
        
        # print(); print(to_drop)

        ting = np.zeros(len(to_drop))
        i = 0
        for j, loop in enumerate(X.columns):
            if(loop == to_drop[i]):
                ting[i] = X_scaled.columns[j]
                i += 1
                if(i == len(ting)):
                    break

        df1 = X.drop(X.columns[ting.astype(int)], axis=1)
        
        # print(); print(df1.head())

        self.data = pd.concat([y, df1], axis=1, join="inner")

    def PCA(self, PCA_n):
        #https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
        SC = StandardScaler()
        X = self.data.drop("Bankrupt?", axis=1)
        y = self.data["Bankrupt?"]

        #Skalerer dataen
        X_scaled = pd.DataFrame(SC.fit_transform(X))
        X = X_scaled.rename(columns={i:j for i,j in zip(X_scaled.columns,X.columns)})

        PCA_data = PCA(n_components=PCA_n)

        PCA_final = PCA_data.fit_transform(X)

        navn = []
        for i in range(PCA_n):
            navn.append("Variable %i"%i)


        Two_PCA = pd.DataFrame(data = PCA_final, columns=navn)

        print(PCA_data.explained_variance_ratio_)

        self.data = pd.concat([y, Two_PCA], axis=1, join="inner")

    def BE(self, p):
        SC = StandardScaler()
        X = self.data.drop("Bankrupt?", axis=1)
        y = self.data["Bankrupt?"]

        #Skalerer dataen
        X_scaled = pd.DataFrame(SC.fit_transform(X))
        X = X_scaled.rename(columns={i:j for i,j in zip(X_scaled.columns,X.columns)})

        OLS_tmp  = sm.OLS(endog = y, exog = X).fit()

        pValues = OLS_tmp.pvalues

        while np.max(pValues) > p:
            X.drop(X.columns[np.where(pValues == np.max(pValues))[0][0]], axis=1, inplace=True)
            OLS_tmp  = sm.OLS(endog = y, exog = X).fit()
            pValues = OLS_tmp.pvalues
        

        self.data = pd.concat([y, X], axis=1, join="inner")