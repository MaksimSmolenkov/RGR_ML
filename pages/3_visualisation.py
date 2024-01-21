import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("classification_data.csv")

def func(x):
    if x == 'Yes':
        return 1
    else:
        return 0

data['Fire Alarm'] = data['Fire Alarm'].apply(func)
data.drop(['Unnamed: 0.1', 'CNT', 'UTC'], axis = 1, inplace = True)
st.dataframe(data)

st.title('Визуализация датасета')

st.header('Датасет для классификации - "Срабатывания датчика дыма"')

st.markdown('---')

st.write("Диаграмма с областями влажности и температуры")

chart_data = pd.DataFrame(
   {
       "Humidity[%]": data.iloc[:, 2].to_numpy(),
       "Temperature": data.iloc[:, 1].to_numpy(),
   }
)
st.area_chart(chart_data)

st.write("Диаграмма рассеиния этанола")

chart_data = pd.DataFrame(data["Raw Ethanol"].to_numpy(), columns=["Raw Ethanol"])
st.scatter_chart(chart_data)


st.write("Гистограмма Распределения температуры")

fig, ax = plt.subplots()

ax.hist(data['Temperature[C]'], bins=20)

st.pyplot(fig)

st.write("Гистограмма Распределения давления")

fig, ax = plt.subplots()

ax.hist(data['Pressure[hPa]'], bins=20)

st.pyplot(fig)

st.write("Гистограмма предсказываемого признака")

fig, ax = plt.subplots()

ax.hist(data['Fire Alarm'], bins=20)

st.pyplot(fig)

st.write("Гистограмма распределения признака H2")
x_values = data['Raw H2']
fig, ax = plt.subplots()
index = list(range(1, 62631))
ax.scatter(x_values, index)

st.pyplot(fig)


st.header("Анализ корреляции между различными признаками набора данных")
fig = plt.figure()
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Тепловая карта корреляции между всеми парами признаков набора данных")
st.write(fig)

