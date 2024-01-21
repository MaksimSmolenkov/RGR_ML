import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('classification_data.csv')


def func(x):
    if x == 'Yes':
        return 1
    else:
        return 0

data['Fire Alarm'] = data['Fire Alarm'].apply(func)
data = data.drop(['Unnamed: 0', 'UTC', 'Unnamed: 0.1'], axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = data.drop('Fire Alarm', axis=1)
df_sc = sc.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=100)
kmeans_fit = kmeans.fit(df_sc)

pickle.dump(kmeans_fit, open('kmeans.pkl', 'wb'))
