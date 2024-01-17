import pandas as pd 
import numpy as np 
import pickle
import math
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import streamlit as st 
import sklearn.metrics 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss 
import io

df = pd.read_csv('classification_data.csv')

def func(x):
    if x == 'Yes':
        return 1
    else:
        return 0

df['Fire Alarm'] = df['Fire Alarm'].apply(func)
df = df.drop(['Unnamed: 0', 'UTC', 'Unnamed: 0.1'], axis=1)



if df is not None:
    st.header("Датасет")
    st.dataframe(df)

    st.write("---")

    st.title("Fire Alarm Prediction") 
    list=[]

    for i in df.columns[:-1]:
        a = st.slider(i,int(df[i].min()), int(math.ceil(df[i].max())),int(df[i].max()/2))
        list.append(a)

    #list.append(a)
    list = np.array(list).reshape(1,-1)
    list=list.tolist()

    st.title("Тип модели обучения")
    model_type = st.selectbox("Выберите тип", ['SVM', 'Kmeans', 'Boosting', 'Bagging','Stacking', 'MLP' ])

    st.markdown('SVM - Support Vector Machine - классическая модель обучения с учителем')
    st.markdown('Kmeans - метод k-средних - алгоритм кластеризации')
    st.markdown('Boosting  - градиентный бустинг - ансамблевая модель')
    st.markdown('Bagging  - бэггинг - ансамблевая модель')
    st.markdown('Stacking  - стекинг - ансамблевая модель')
    st.markdown('MLP  - multy layer perceptron(многослойный персептрон) - полносвязная нейронная сеть')



    

    button_clicked = st.button("Предсказать")
    if button_clicked:
        if model_type is not None:
            if model_type == "SVM":
                with open('models/clf.pkl', 'rb') as file:
                    knn_model = pickle.load(file)
                if knn_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                elif knn_model.predict(list) == 1:
                    st.success("Пожарная тревога будет")

            elif model_type == "Kmeans":
                with open('models/kmeans.pkl', 'rb') as file:
                    kmeans_model = pickle.load(file)
                if kmeans_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                elif kmeans_model.predict(list) == 1:
                    st.success("Пожарная тревога будет")

            elif model_type == "Boosting":
                with open('models/gradient_booster.pkl', 'rb') as file:
                    boos_model = pickle.load(file)
                if boos_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                elif boos_model.predict(list) == 1:
                    st.success("Пожарная тревога будет")

            elif model_type == "Bagging":
                with open('models/bagging.pkl', 'rb') as file:
                    bagg_model = pickle.load(file)
                if bagg_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "Stacking":
                with open('models/stacking.pkl', 'rb') as file:
                    stac_model = pickle.load(file)
                if stac_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")

            elif model_type == "MLP":
                with open('models/mlp.pkl', 'rb') as file:
                    mlp_model = pickle.load(file)
                if mlp_model.predict(list) == 0:
                    st.success("Пожарной тревоги не будет")
                else:
                    st.success("Пожарная тревога будет")
