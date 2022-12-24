#!/usr/bin/env python
# coding: utf-8
# %%

# %%

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plost
import base64


st.set_page_config(page_title= "PROYECTO DE INVESTIGACIÓN FORMATIVA")
def Grupo1_5():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
    st.title("PROYECTO DE INVESTIGACIÓN FORMATIVA")

st.header('Universidad Nacional de San Agustín de Arequipa') 
st.header("Escuela Profesional de Ingeniería de Telecomunicaciones")
st.image("https://www.unsa.edu.pe/wp-content/uploads/sites/3/2018/05/Logo-UNSA.png",
width=300,)
st.subheader('Docente : Ingeniero Renzo Bolivar')
st.subheader("Curso : Computación 1")
st.subheader("GRUPO C - Nº5")
st.subheader('Integrantes:') 
st.text("Lope Condori Santiago Isaac")
st.text("Montalvo Pacori Ivan")
st.text("Ramos Catari Joaquin")
st.text("Quispe Coila Yampier Edison")
st.text("Vilca Medina Milagros Mercedes")

def Correlación():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")
st.header("CORRELACIÓN")
#archivo CSV separado por comas

data = pd.read_csv("Lugares.csv")

#leer  lineas
data


# %%
st.header("DATA.SHAPE")
data.shape


# %%
st.header("DATA.DTYPES")
data.dtypes



# %%
st.header("IMPUTACIÓN DE VALORES NULOS") 
data1= data.fillna(data.mean())
data1




st.image("https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png")



# %%

                       
n = data1[data1.columns[1:]].to_numpy()
m = data1[data1.columns[0]].to_numpy()
print(n)
print(m)




# %%
st.header("CORRELACIÓN EN PANDAS") 

n.T


# %%


df1 = pd.DataFrame(n.T, columns = m)
df1


# %%


m_corr = df1.corr()
m_corr


st.image("https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png")



# %%
st.header("MATRIX DE CORRELACIÓN") 

m_corr_pandas = np.round(m_corr, 
                       decimals = 2)  
  
m_corr_pandas


# %%


("m_corr_pandas")


# %%


pandas= m_corr_pandas.unstack()

print(pandas.sort_values(ascending=False)[range(len(n),((len(n)+4)))])


# ## Gráfica de Calor



# %%
st.header("GRÁFICA DE CALOR 1")
pd.read_csv("Lugares.csv")
m_corr_pandas=df1.corr()
mask=np.zeros_like(m_corr_pandas)
mask[np.triu_indices_from(mask)]= True
with sns.axes_style("white"):
    f, ax =plt.subplots(figsize=(5,5))
    ax= sns.heatmap(m_corr_pandas, mask=mask, vmax=1, square=True)
        
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


# %%

# %%
st.header("5.- RESULTADOS") 

st.subheader("Los resultados de similitud obtenidos en **Lugares que me gustaría viajar** según la tabla de **Correlación** con los siguientes encuestados:")
st.caption("1. cosmic99latte@gmail.com y milam@gmail.com  obtienen el **PRIMER** indice mas alto de similitud con el 72% ")
st.caption("2. esmelizeth@gmail.com   y  milam@gmail.com obtienen el **SEGUNDO** indice mas alto de similitud con el 70%")





# %%


n = data1[data1.columns[1:]].to_numpy()
m = data1[data1.columns[0]].to_numpy()
print(n)
print(m)


# %%
st.header("Validación - Matrix de Correlación")

import math
corr_grupal=[]

def correlaciongrupal(x,y):
    xprom, yprom=x.mean(), y.mean()
    arriba=np.sum((x-xprom)*(y-yprom))
    abajo=np.sqrt(np.sum((x-xprom)**2)*np.sum((y-yprom)**2))
    return arriba/abajo

for columna in range (len(m)):
        for fila in range(len(m)):
            datos=data1.loc[[columna,fila],:]
            datos2=datos[datos.columns[1:]].to_numpy()
            corr_grupal.append(correlaciongrupal(datos2[0],datos2[1]))
            
corre_grupal=np.array(corr_grupal).reshape(len(m),len(m))
correlacion=pd.DataFrame(corre_grupal,m,m)
correlacion
    


# %%


pandas= correlacion.unstack()

print(pandas.sort_values(ascending=False)[range(len(n),((len(n)+4)))])


# %%


import matplotlib.pyplot as plt
st.header("GRÁFICA DE CALOR 2")
pd.read_csv("Lugares.csv")
m_corr_pandas=df1.corr()
mask=np.zeros_like(m_corr_pandas)
mask[np.triu_indices_from(mask)]= True
with sns.axes_style("white"):
    f, ax =plt.subplots(figsize=(5,5))
    ax= sns.heatmap(m_corr_pandas, mask=mask, vmax=1, square=True)
        
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

page_names_to_funcs = {
    "Presentacion": Grupo1_5,
    "Marco Teórico": Marco_Teórico,
    "Correlación": Correlación,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

# %%
