from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import seaborn as sns
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

gradient_booster = GradientBoostingClassifier(learning_rate=0.1)

gradient_booster.fit(X_train,y_train)

pickle.dump(gradient_booster, open('gradient_booster.pkl', 'wb'))

