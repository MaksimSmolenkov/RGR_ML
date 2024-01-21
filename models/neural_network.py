import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

classification_data = pd.read_csv('classification_data.csv')

def func(x):
    if x == 'Yes':
        return 1
    else:
        return 0

classification_data['Fire Alarm'] = classification_data['Fire Alarm'].apply(func)
classification_data = classification_data.drop(['Unnamed: 0', 'UTC', 'Unnamed: 0.1'], axis=1)

cdf= classification_data

y_classification = cdf['Fire Alarm']
X_classification = cdf.drop(columns = ['Fire Alarm'])

X_classification_train, X_classification_test, y_classification_train, y_classification_test = train_test_split(X_classification,
                                                                                                                y_classification,
                                                                                                                stratify=y_classification,
                                                                                                                test_size=0.2)

scaler = StandardScaler()
data_scaler = scaler.fit_transform(cdf.drop(["Fire Alarm"], axis=1))

y_classifier = cdf["Fire Alarm"]
X_classifier = data_scaler


X_train_classifier, X_test_classifier, y_train_classifier, y_test_classifier = train_test_split(X_classifier, y_classifier, test_size=0.2)

model_classification = MLPClassifier(solver='adam', alpha=0.00000001,
                     n_iter_no_change = 200, random_state=3)

model_classification.fit(X_classification_train, y_classification_train)

pickle.dump(model_classification, open('model.pkl', 'wb'))