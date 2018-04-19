import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
balance_data = pd.read_csv(
'C:\\Users\\akash\\Desktop\\sonar.csv',
                           sep= ',', header= None)
#Getting the dataset
print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)
X = balance_data.values[:, 0:59]#features
Y = balance_data.values[:,60]#label
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
#using Gaussian naives bayes
clf_gini =GaussianNB() 
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print(y_pred)
print(y_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
#using Bernoulli naives bayes
clf_gini =BernoulliNB()
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print(y_pred)
print(y_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
#using Multinomial naives bayes
clf_gini =MultinomialNB()
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print(y_pred)
print(y_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
