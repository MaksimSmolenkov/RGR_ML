from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from mlxtend.classifier import StackingClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pickle
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.model_selection import cross_val_score
sns.set()
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('classification_data.csv')

def func(x):
    if x == 'Yes':
        return 1
    else:
        return 0

df['Fire Alarm'] = df['Fire Alarm'].apply(func)
df = df.drop(['Unnamed: 0', 'UTC', 'Unnamed: 0.1'], axis=1)


y = df['Fire Alarm']
X = df.drop('Fire Alarm', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=lr)

params = {'kneighborsclassifier__n_neighbors': [1, 5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta_classifier__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf,
                    param_grid=params,
                    cv=5,
                    refit=True)
grid.fit(X_train, y_train)

pickle.dump(grid, open('stacking.pkl', 'wb'))
