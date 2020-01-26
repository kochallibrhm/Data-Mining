# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 01:30:54 2020

@author: kochalilibrahim
"""
# pandas kutuphanesini import ediyoruz (importing pandas library for read the data)
import pandas as pd


# Veri yukleme: .csv uzantili hale getirdigimiz veri kumemizi pandas kutuhanesi kullanarak yukluyoruz
# Reading the data with pandas
veriler = pd.read_csv("Proje1VeriKumesi.csv", encoding='utf-8')
classless = pd.read_csv("Proje1Sinifsiz.csv", encoding='utf-8')

# Veri Onisleme (Data preprocessing)
# Eksik verileri doldurmak icin sütunların ortalamasını kullanıyoruz.
# We are using the mean for missing values.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

x = veriler.iloc[:, 1:10].values  # bagimsiz degiskenler (independent variables)
y = veriler.iloc[:, 10:].values  # bagimli degisken (dependent variable)
xClassless = classless.iloc[:, 1:10].values  # bagimsiz degisken classless (independent variables for classles data)

imputer = imputer.fit(x[:, 1:10])
x[:, 1:10] = imputer.transform(x[:, 1:10])

# Verilerin egitim ve test icin bölünmesi, verilerin 1/3 lük kismini test icin ayiriyoruz. 
# Seperating  the data to 3 piece for training and test. 1 out of 3 for test, 2 out of 3 for training.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# verileri ölcekliyoruz. (scaling transaction for data)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
xClassless = sc.transform(xClassless)

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
knn.fit(X_train, y_train)  # Training

y_pred = knn.predict(X_test)  # predicts for test class
predClassless = knn.predict(xClassless)  # predicts for classless data

# Confusion matrix for KNN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("---------------------------------KNN------------------------------------------\n")
print("Confusion Matrix: ")
print(cm)
print("\nKNN predicts for classless datas: ")
print(" ID = 24: "+predClassless[0] + ", ID = 44: "+predClassless[1] + ", ID = 54: "+predClassless[2]+"\n")


# Desicion Tree
from sklearn import tree
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)  # Training

y_pred = dtc.predict(X_test)  # predicts for test class
predClassless = dtc.predict(xClassless)  # predicts for classless data

# Confusion Matrix for DSCTREE
cm = confusion_matrix(y_test, y_pred)
print("--------------------------------DSCTREE----------------------------------------\n")
print("Confusion Matrix: ")
print(cm)
print("\nDSCTREE predicts for classless datas: ")
print(" ID = 24: "+predClassless[0] + ", ID = 44: "+predClassless[1] + ", ID = 54: "+predClassless[2]+"\n")

# Neural Network Models Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5)
mlp.fit(X_train, y_train)  # Training

y_pred = mlp.predict(X_test)  # predicts for test class
predClassless = mlp.predict(xClassless)  # predicts for classless data

# Confusion Matrix for MLP
cm = confusion_matrix(y_test, y_pred)  
print("---------------------------NEURAL_NETWORK_MLP----------------------------------\n")
print("Confusion Matrix: ")
print(cm)
print("\nNEURAL_NETWORK_MLP predicts for classless datas: ")
print(" ID = 24: "+predClassless[0] + ", ID = 44: "+predClassless[1] + ", ID = 54: "+predClassless[2]+"\n")















