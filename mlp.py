import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

df = pd.read_csv('autism_screening.csv') 
# print(df.shape)
# df.describe().transpose()

target_column = ['Class/ASD'] 
predictors = ['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score']
df[predictors] = df[predictors]/df[predictors].max()
# df.describe().transpose()

X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=45)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(5,8,5), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train,y_train.ravel())
pickle.dump(mlp,open('qqq.pkl','wb'))

# from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_train,predict_train))
# print(classification_report(y_train,predict_train))
