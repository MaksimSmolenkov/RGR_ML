import pandas as pd
import numpy as np
from sklearn import svm
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('classification_data.csv')

def func(x):
    if x == 'Yes':
        return 1
    else:
        return 0

data['Fire Alarm'] = data['Fire Alarm'].apply(func)
data = data.drop(['Unnamed: 0', 'UTC', 'Unnamed: 0.1'], axis=1)

y = data['Fire Alarm']
X = data.drop('Fire Alarm', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

pickle.dump(clf, open('clf.pkl', 'wb'))


