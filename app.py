import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.decomposition import PCA
import sklearn.neighbors
from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import DBSCAN

import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings('ignore')
import io


def main():
    st.set_page_config(layout='wide')
    
    # Crear sidebar con opciones
    #st.sidebar.header('Seleccionar Opcion:')
    
    # Lista de opciones del sidebar
    opciones = ['Seleccionar...', 'Python','Numpy','Pandas','Matplotlib','Seaborn','Plotly','Bokeh']
    
    opcion_seleccionada = st.sidebar.selectbox(
        '**Bibliotecas:**',
        opciones
    )
    
    st.sidebar.write('---')
    
    # Lista de opciones del sidebar
    opciones_2 = ['Seleccionar...', 'Introducción', 'Regresión Lineal','Regresión Logística','KNN, K vecinos más cercanos', 'Arbol de Decisión y Bosque Aleatorio',
'Conjunto de Arboles de Desicion','Maquina de Soporte Vectorial', 
'Analisis de Componentes Principales (PCA)','Sistemas de Recomendación','Procesamiento de Lenguaje Natural (NLP)', 'No Supervisado - Clustering','Bayes Ingenuo']
    
    opcion_seleccionada_2 = st.sidebar.selectbox(
        '**Machine Learning:**',
        opciones_2
    )
    
    st.sidebar.write('---')    
    
    opciones_3 = ['Seleccionar...','Git y GitHub','AWS','Docker','Streamlit']
 
    opcion_seleccionada_3 = st.sidebar.selectbox(
        '**Herramientas:**',
        opciones_3
    )
    
    
    st.sidebar.info('Ultima Actualización: 27/11/2025')
    st.sidebar.success('App en constante actualización.')
    
    if opcion_seleccionada != 'Seleccionar...':
        st.title(opcion_seleccionada)
    
        if opcion_seleccionada == 'Python':
            python()
        elif opcion_seleccionada == 'Numpy':
            numpy()
        elif opcion_seleccionada == 'Pandas':
            pandas()
        elif opcion_seleccionada == 'Matplotlib':
            matplotlib()
        elif opcion_seleccionada == 'Seaborn':
            seaborn()
            
    if opcion_seleccionada_2 != 'Seleccionar...':
        st.title(opcion_seleccionada_2)
        
        if opcion_seleccionada_2 == 'Introducción':
            ml_introduccion()        
        elif opcion_seleccionada_2 == 'No Supervisado - Clustering':    
            ml_aprendizaje_NoSupervisado()
        elif opcion_seleccionada_2 == 'Regresión Lineal': 
            ml_regresion_lineal()
        elif opcion_seleccionada_2 == 'Regresión Logística':
            ml_regresion_logistica()
        elif opcion_seleccionada_2 == 'KNN, K vecinos más cercanos':
            ml_knn()
        elif opcion_seleccionada_2 == 'Arbol de Decisión y Bosque Aleatorio':
            ml_trees()
        elif opcion_seleccionada_2 == 'Conjunto de Arboles de Desicion':
            ml_ensambletrees()
        elif opcion_seleccionada_2 == 'Maquina de Soporte Vectorial':
            ml_SVM()
        elif opcion_seleccionada_2 == 'Analisis de Componentes Principales (PCA)':
            ml_PCA()
        elif opcion_seleccionada_2 == 'Bayes Ingenuo':
            ml_Bayes()
        elif opcion_seleccionada_2 == 'Sistemas de Recomendación':
            ml_SisRecomendacion()
        elif opcion_seleccionada_2 == 'Procesamiento de Lenguaje Natural (NLP)':
            ml_NLP()
            
    if opcion_seleccionada_3 != 'Seleccionar...':
        st.title(opcion_seleccionada_3)
        
        if opcion_seleccionada_3 == 'Git y GitHub':
            git()
        elif opcion_seleccionada_3 == 'Docker':
            docker()
        elif opcion_seleccionada_3 == 'AWS':
            aws()
        
        
def ml_introduccion():
        import matplotlib.pyplot as plt 
        opciones_mlmodleado = ['Machine Learning', 'Analisis de Datos (Diabetes)', 'Analisis de Datos (Countries)', 'Procesamiento de Datos']
    
        col1, col2 = st.columns([2,2])
    
        with col1:
                opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlmodleado)
                st.success(f'##### **{opcion_seleccionada}** ')
    

        if opcion_seleccionada == 'Procesamiento de Datos':
                st.write('##### Definición')
                st.write('''El Procesamiento de Datos transforma los datos brutos en informacion estructurada para entrenar modelos, lo que implica limpieza, normalizacion, manejo de valores nulos y codificaicon
de variables categoricas.''')        
    
    
                st.write('''Los pasos clave del procesamiento de datos incluyen:
                         
**Recopilación de datos**: Obtención de fuentes estructuradas o no estructuradas (CSV, bases de datos).   
**Limpieza de datos**: Tratamiento de datos faltantes, eliminación de duplicados y gestión de valores atípicos.   
**Ingeniería de características (Feature Engineering)**: Selección, creación o transformación de variables para mejorar la capacidad predictiva.  
**Normalización y Escalado**: Ajustar los valores numéricos a una escala común (ej. entre 0 y 1) para que características con mayores magnitudes no dominen el modelo.    
**Codificación de categóricas**: Conversión de variables de texto en números (ej. one-hot encoding) para que los algoritmos puedan procesarlas. ''')
    

                st.code('''columns =  ['preg','plas','pres','skin','test','mass','pedi','age','class']
df_diabetes = pd.read_csv('DataFrames/pima-indians-diabetes.csv', names=columns)
df_diabetes''')
                
                columns =  ['preg','plas','pres','skin','test','mass','pedi','age','class']
                df_diabetes = pd.read_csv('DataFrames/pima-indians-diabetes.csv', names=columns)

                st.dataframe(df_diabetes.head(10))    
    
                st.divider()
                st.write('##### Metodos de Transformacion de datos')
                
                st.write('''**Escalamiento**: esta transformacion es util para los algoritmos de optimizacion utilizados en el nucleo de los algoritmos de aprendizaje automatico como Gradiente Descendiente.  
Tambien es util para algoritmos que ponderan entradas como Regression y Neural Networks y algoritmos que usan medidas de distancia como k-Nearest Neighbours.   
Puede reescalar sus datos usando la clase MinMaxScaler. Despues de reescalar puede ver que todos los valores estan en el rango [0,1]''')
    
    
                st.code('''# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_diabetes = scaler.fit_transform(df_diabetes)''')

                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler(feature_range=(0,1))
                rescaled_diabetes = scaler.fit_transform(df_diabetes)
                rescaled_diabetes
                
                
                st.write('''**Estandarizacion**: Es mas adecuada para tecnicas que asumen una distribucion gausssiana en las variables de entrada y funcionan mejor con datos reescalados, como LiR,
LoR y LDA.      
Puede estandarizar datos utilizando la clase StandardScaler. Los valores para cada atributo ahora tienen un valor medio de 0 y una desviacion estandar de 1.''')
                
                st.code('''# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df_diabetes)
rescaled_diabetes = scaler.transform(df_diabetes)''')                
                
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler().fit(df_diabetes)
                rescaled_diabetes = scaler.transform(df_diabetes)
                rescaled_diabetes
    
        
                st.write('''**Normalizacion**: los valores de los datos se pueden escalar en el rango de [0,1]. Este metodo de preprocesamiento puede ser util para conjuntos de datos dispersos
(muchos ceros) con atributos de escalas variables.      
Cuando se utilizan algoritmos que ponderan valores de entrada como NN y algoritmos que usan medidas de distancia como k-NN. Para normalizar datos se utiliza la clase Normalizer.''')

                st.code('''# Normalizer
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(df_diabetes)
rescaled_diabetes = scaler.transform(df_diabetes)''')                
                
                from sklearn.preprocessing import Normalizer

                scaler = Normalizer().fit(df_diabetes)
                rescaled_diabetes = scaler.transform(df_diabetes)
                rescaled_diabetes        

                st.write('''**Binarizacion**: puede crear nuevos atributos binarios en Python usando la clase Binarizer.        
Puede ver que todos los valores iguales o menores que 0 estan marcados con 0 y todos los que estan por encima de 0 estan marcados con 1.''')
        
        
                st.code('''# Binarizer
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(df_diabetes)
binary_diabetes = binarizer.transform(df_diabetes)''')                
                
                from sklearn.preprocessing import Binarizer

                binarizer = Binarizer(threshold=0.0).fit(df_diabetes)
                binary_diabetes = binarizer.transform(df_diabetes)     
                binary_diabetes
        
        
                st.write('''**Box-Cox**: los atributos representan un sesgo o inclinacion (Gaussiana desplazada).       
Box-Cox asume todos los atributos positivos. Aplica la transformacion a los atributos que parecen tener sesgo.  
Corrige la no linealidad en la relacion (mejorar correlacion entre las variables.)''')
        
                st.write('''**Yeo-Johnson**: igual que Box-Cox pero soporta valores en bruto que son iguales a cero y negativos.''')        
        
                st.divider()
                st.write('##### Metodos de remuestreo')        
        
                st.write('''**Objetivo**: Evaluar los algoritmos.       
Dividir los datos para la evaluacion. Existen 4 enfoques de division:   
* Como dividir un conjunto de datos en subconjuntos por porcentaje para entrenamiento/validacion.
* Como evaluar la robustez del modelo utilizando la validacion cruzada, k-fold, con y sin repeticiones.
* Como evaluar la robustez del modelo usando una validacion cruzada dejando uno afuera (LOOCV).
Division en train/test repetidos aleatoriamente.''')
        
        
        if opcion_seleccionada == 'Analisis de Datos (Countries)':
                st.write('##### Definición')
                st.write('''El Analisis Exploratorio de Datos en machine learning es un enfoque critico para analizar conjuntos de datos y resuminr sus caracteristicas principales, utilizando metodos estadisticos
y visualizacion de datos.       
Permite a los cientificos de datos entender la estructura de datos, detectar anomalias, probar suposiciones y encontrar patrones ocultos antes del modelado formal.''')
        
                st.divider()
                st.write('##### Carga de datos en un dataset')
                
                st.write('''**read_csv()** : Es una funcion que se utiliza para importar archivos CSV a un DataFrame. Lee los datos, y separa los valores por comas por defecto.   

**Parametros**  

* sep: el caracter utilizado para separar los valores (delimitador). El predeterminado es la coma ','.    
* header: la fila que se usará como encabezado, header=0 (primera fila) o header=None.    
* names: una lista de nombres de columna para usar en caso de que el archivo no tenga encabezado. 
* index_col: especifica la columna a usar como índice del DataFrame.  
* na_values: se utiliza para especificar que valores deben interpretarse como valores faltantes (NaN) al cargarlo en un DataFrame.  
Se pueden pasar una lista de cadenas (n/a, ---, ?, etc.) ademas de los valores predeterminados como '', 'NULL', 'NA', etc.''')
                
                
                st.code('''df_countries = pd.read_csv('DataFrames/countries.csv', sep=';')''')        
                df_countries = pd.read_csv('DataFrames/countries.csv', sep=';')
                
                st.divider()
                st.write('''**shape** : se utiliza para conocer las dimensiones de un DataFrame o Serie, devolviendo una tupla con la estrucutra 
        (numero de filas, numero de columnas)''')

                st.code('''df_countries.shape''')
                df_countries.shape
        
                st.divider()
                st.write('''**columns** : se utiliza para visualizar o modificar los nombres de las columnas en un DataFrame, devolviendo un objeto de tipo indice (Index). 
        Es fundamental para la manipulacion de datos tabulares, permitiendo renombrar, seleccionar, añadir o eliminar columnas.''')

                st.code('''df_countries.columns''')
                st.code(df_countries.columns)

                st.divider()
                st.write('''**head()** : se utiliza para visualizar rapidamente las primeras filas de un DataFrame o Series, siendo el valor 5 el valor predeterminado.''')

                st.code('''df_countries.head(10)''')
                st.code(df_countries.head(10), language='html')


                st.divider()
                st.write('''**info()** : proporciona un resumen conciso y esencial de un DataFrame, mostrando el numero de filas (entradas), nombres de columnas, tipos de datos (dtypes),  
        valores no nulos y el uso total de memoria.''')

                st.code('''df_countries.info()''')

                buffer = io.StringIO()   
                df_countries.info(buf=buffer)             
                st.code(buffer.getvalue(), language='html')  

                
                st.divider()        
        
                st.write('##### Ver valores nulos')
                st.write('''**isnull().sum()** : cuenta el numero total de nulos por columna.''')     
                st.code('''df_countries.isnull().sum()''')          
                st.code(df_countries.isnull().sum(), language='html')  
        
                st.divider()
                
                # Matriz de correlacion 

                # Tomar valores numericos
                st.write('**Tomar valores numericos**')
                st.code('''df_countries_numericos = df_countries.select_dtypes(include=['float64','int64'])
df_countries_numericos.head()''')
                
                df_countries_numericos = df_countries.select_dtypes(include=['float64','int64'])
                st.dataframe(df_countries_numericos.head())
        
        
                st.write('**Matriz de correlacion**')
                st.code('''correlacion = df_countries_numericos.corr(method='pearson')
correlacion''')
        
                correlacion = df_countries_numericos.corr(method='pearson')
                correlacion

                st.code('''fig, ax = plt.subplots()
cax = ax.matshow(correlacion, vmin=-1, vmax=1)
fig.colorbar(cax)
st.pyplot(fig) ''')
                
                with st.container(width=600):
                        fig, ax = plt.subplots()
                        cax = ax.matshow(correlacion, vmin=-1, vmax=1)
                        fig.colorbar(cax)       
                        
                        st.pyplot(fig) 
                
                st.write('**heatmap()**')
                st.code('''fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')
plt.title('Correlacion entre variables.')''')
                
                with st.container(width=800):
                        fig, ax = plt.subplots(figsize=(10,10))
                        sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')   
                        plt.title('Correlacion entre variables.')
                        st.pyplot(fig) 
        
                st.divider()


                st.write('##### Deteccion y analisis de outliers')






























        
        
        
        
        
        
        if opcion_seleccionada == 'Analisis de Datos (Diabetes)':
                st.write('##### Definición')
                st.write('''El Analisis Exploratorio de Datos en machine learning es un enfoque critico para analizar conjuntos de datos y resuminr sus caracteristicas principales, utilizando metodos estadisticos
y visualizacion de datos.       
Permite a los cientificos de datos entender la estructura de datos, detectar anomalias, probar suposiciones y encontrar patrones ocultos antes del modelado formal.''')
        
                st.divider()
                st.write('##### Carga de datos en un dataset')
                
                st.write('''**read_csv()** : Es una funcion que se utiliza para importar archivos CSV a un DataFrame. Lee los datos, y separa los valores por comas por defecto.   

**Parametros**  

* sep: el caracter utilizado para separar los valores (delimitador). El predeterminado es la coma ','.    
* header: la fila que se usará como encabezado, header=0 (primera fila) o header=None.    
* names: una lista de nombres de columna para usar en caso de que el archivo no tenga encabezado. 
* index_col: especifica la columna a usar como índice del DataFrame.  
* na_values: se utiliza para especificar que valores deben interpretarse como valores faltantes (NaN) al cargarlo en un DataFrame.  
Se pueden pasar una lista de cadenas (n/a, ---, ?, etc.) ademas de los valores predeterminados como '', 'NULL', 'NA', etc.''')
                st.code('''columns =  ['preg','plas','pres','skin','test','mass','pedi','age','class']''')
                st.code('''df_diabetes = pd.read_csv('DataFrames/prima-indians-diabetes.csv', names=columns)''')
                
                columns =  ['preg','plas','pres','skin','test','mass','pedi','age','class']
                df_diabetes = pd.read_csv('DataFrames/pima-indians-diabetes.csv', names=columns)
                

                st.divider()
                st.write('''**shape** : se utiliza para conocer las dimensiones de un DataFrame o Serie, devolviendo una tupla con la estrucutra 
        (numero de filas, numero de columnas)''')

                st.code('''df_diabetes.shape''')
                df_diabetes.shape
        
                st.divider()
                st.write('''**columns** : se utiliza para visualizar o modificar los nombres de las columnas en un DataFrame, devolviendo un objeto de tipo indice (Index). 
        Es fundamental para la manipulacion de datos tabulares, permitiendo renombrar, seleccionar, añadir o eliminar columnas.''')

                st.code('''df_diabetes.columns''')
                st.code(df_diabetes.columns)

                st.divider()
                st.write('''**head()** : se utiliza para visualizar rapidamente las primeras filas de un DataFrame o Series, siendo el valor 5 el valor predeterminado.''')

                st.code('''df_diabetes.head(20)''')
                st.code(df_diabetes.head(20), language='html')
                
                st.divider()
                        
                st.write('''**dtypes** : devuelve una serie con el tipo de datos de cada columna en un DataFrame.''')       
                st.code('''df_diabetes.dtypes''')
                st.code(df_diabetes.dtypes, language='html')
        

                
                st.divider()
                st.write('''**info()** : proporciona un resumen conciso y esencial de un DataFrame, mostrando el numero de filas (entradas), nombres de columnas, tipos de datos (dtypes),  
        valores no nulos y el uso total de memoria.''')

                st.code('''df_diabetes.info()''')

                buffer = io.StringIO()   
                df_diabetes.info(buf=buffer)             
                st.code(buffer.getvalue(), language='html')    
                
                
                st.divider()
                st.write('''**describe()** : funcion esencial para el analisis exploratorio de datos, que genera un resumen estadistico descriptivo de las columnas numericas en un DataFrame.  
        Proporciona metricas clave como el conteo (count), media (mean), desviacion estandar (std), valores minimos/maximos y percentiles (25%, 50%, 75%)''')        
                
                st.code('''df_diabetes.describe().T''')
                st.code(df_diabetes.describe().T, language='html')
                
                st.divider()
        
        
        
                st.write('''**groupby().size()** : el metodo size() utilizado tras un groupby() devuelve el numero de filas o elementos en cada grupo como una Serie.   
        A diferencia de count(), incluye valores NaN (nulos) en el conteo total. Se utiliza principalmente para obtener la frecuencia de ocurrencia de cada grupo.''')       
                st.code('''df_diabetes.groupy('class').size()''')
                st.code(df_diabetes.groupby('class').size(), language='html')                
                        
                st.divider()
        
                st.write('##### Correlaciones')
                st.write('''Se define la relacion entre pares de atributos numericos. Los valores superiores a aprox. 0.75 e inferiores a -0.75 son los mas interesantes ya que muestran una alta correlacion.  
        1 y -1 correlacion positiva o negativa completa.''')
                st.write('''**corr()** : metodo que calcula la matriz de correlacion de Pearson para columans numericas, mostrando relaciones lineales entre -1 y 1.
        Ignora los valores no numericos y es esencial para analisis exploratorio de datos, permitiendo identificar como varian conjuntamente las variables.''')        
                
                st.code('''correlacion = df_diabetes.corr(method='pearson') ''')
                correlacion = df_diabetes.corr(method='pearson')        
        
        
                st.write('##### Matriz de correlacion')
                
                st.code('''fig, ax = plt.subplots()
cax = ax.matshow(correlacion, vmin=-1, vmax=1)
fig.colorbar(cax)   
plt.show()''')

                        
                with st.container(width=800):
                        fig, ax = plt.subplots()
                        cax = ax.matshow(correlacion, vmin=-1, vmax=1)
                        fig.colorbar(cax)       
                        
                        st.pyplot(fig) 
                
                st.write('**heatmap()**')
                st.code('''fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')  ''')
                
                with st.container(width=800):
                        fig, ax = plt.subplots(figsize=(10,10))
                        sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')   

                        st.pyplot(fig) 
        
                st.divider()
                
                st.write('##### Matriz de dispersion')
                st.write('Es util para mirar las relaciones por partes desde diferentes perspectivas.')
                
                st.write('''**scatter_matrix**: muestra una matriz de graficos de dispersion cruzando las caracteristicas cuantitativas del dataframe indicado.''')
                
                st.code('''from pandas.plotting import scatter_matrix

fig, ax = plt.subplots(figsize=(18,18))
scatter_matrix(df_diabetes, ax=ax)
st.pyplot(fig)''')
        
                from pandas.plotting import scatter_matrix

                
                with st.container(width=1200):
                        fig, ax = plt.subplots(figsize=(20,20)) 
                        scatter_matrix(df_diabetes, ax=ax)
                        st.pyplot(fig)
                 
                
                st.write('**sns.pairplot()**')
                st.code('''fig = sns.pairplot(df_diabetes)
st.pyplot(fig)''')
                
                with st.container(width=1200):
                        fig = sns.pairplot(df_diabetes)
                        st.pyplot(fig)
                
                st.write('**Pairplot por clase**')
                st.code('''fig = sns.pairplot(df_diabetes, hue='class, diag_kind='hist')
st.pyplot(fig) ''')
                
                with st.container(width=1200):
                        fig = sns.pairplot(df_diabetes, hue = 'class', diag_kind='hist')
                        st.pyplot(fig) 
        
        
                st.divider()
                st.write('##### Asimetria')
                st.write('''Si una distribucion parece casi gaussiana pero se empuja hacia la izquiera o hacia la derecha, es util conocer el sesgo. 
        Valores cercanos a cero tienen un menor sesgo, sin embargo, sesgo a la izquierda sera con valores negativos y sesgo a la derecha seran valores positivos.''')   
                st.write('''**skew()** : metodo que calcula la asimetria (sesgo) de la distribucion de los datos en un DataFrame o Serie, indicando so los datos son simetricos o estan inclinados hacia un lado respecto a la media.
        Un valor cercano a cero indica simetria, positivo sigifica cola a la derecha, y negativo, cola a la izquierda.''')        
                
                st.code('''df_diabetes.skew()''')   
                st.code(df_diabetes.skew(), language='htmls')
                
                st.divider()
                st.write('##### Ver si existen valores nulos.')
                
                st.write('''**isnull()** : funcion que se utiliza para detectar valores faltantes (NaN, None, NaT) en un DataFrame o Series, devolviendo un objeto booleano del mismo tamaño    
        donde True indica un valor nulo y False uno valido.''')
                
                st.code('''df_diabetes.isnull()''')
                st.code(df_diabetes.isnull(), language='html')
                
                st.write('''**isnull().any()** : se utiliza para detectar rapidamente si alguma columna de un DataFrame contiene valores nulos (Nan, None), devolviendo una Serie booleana (True/False) por columna.''')
                st.code('''df_diabetes.isnull().any()''')
                st.code(df_diabetes.isnull().any(), language='html')     
        
                st.write('''**isnull().any().any()** : comprueba si hay cualquier valor nulo en todo el DataFrame (devuelve un unico booleano)''')     
                st.code('''df_diabetes.isnull().any().any()''')
                st.code(df_diabetes.isnull().any().any(), language='html')        

                st.write('''**isnull().sum()** : cuenta el numero total de nulos por columna.''')     
                st.code('''df_diabetes.isnull().sum()''')          
                st.code(df_diabetes.isnull().sum(), language='html')  
                

                st.divider()
                st.write('##### Valores unicos.')

                st.write('''**unique()** : permite saber cuales son valores unicos de una columna''')
                st.code('''df_diabetes['class'].unique()''')
                st.code(df_diabetes['class'].unique())

                st.write('''**nunique()** : devuelve la cantidad de valores unicos''')
                st.code('''df_diabetes['class'].nunique()''')
                st.code(df_diabetes['class'].nunique())


                st.divider()
                st.write('##### Correccion de Inconsistencias.')

                st.write('''**map()** : permite substituir cada valor de una columna por otro valor basandose en un diccionario, una funcion u otra columna.''')

                st.code('''data = {0:'No Tiene', 1:'Tiene'}  
                        
df_diabetes['class_texto'] = df_diabetes['class'].map(data)''')
                
                data = {0:'No Tiene', 1:'Tiene'} 
                df_diabetes['class_texto'] = df_diabetes['class'].map(data)

                st.code(df_diabetes.head(), language='html')

                st.divider()
                st.write('##### Reemplazar valores nulos.')

                st.write('''**fillna()** : remplaza los valores NaN por otro valor.''')

                st.code('''df_diabetes['test'] = df_diabetes['test'].fillna(0)''')


                st.write('Si no se establece una columna entonces reemplaza las de todas.')
                st.code('''df_diabetes.fillna(0, inplace=True)''')
        # df_creditos.fillna(0, inplace=True)

                st.divider()
                st.write('##### Rvision de valores.')

                st.write('''**value_counts()** : obtiene la cantidad de una variable agrupada por categoría.''')
                st.code('''df_diabetes['class'].value_counts()''')
                st.code(df_diabetes['class'].value_counts(), language='html')        
        
                st.divider()
                st.write('''**replace()** : permite reemplazar los valores por otra etiqueta o crear una nueva columna en base a los valores de la otra columna.''')
                st.code('''df_diabetes['class_Bool'] = df_diabetes['class'].replace({0:False, 1:True})''')
                df_diabetes['class_Bool'] = df_diabetes['class'].replace({0:False, 1:True})
                
                st.code(df_diabetes.head(), language='html')
                
                st.divider()
                st.write('''**aply()** : Es una herramienta de propósito general para aplicar una función a lo largo de un eje (filas o columnas) de un DataFrame o a cada elemento de una Serie.       
        Se utiliza para realizar transformaciones, cálculos y lógica condicional compleja, siendo una alternativa más eficiente y limpia que los bucles.        
        Para usarlo, se le pasa la función a aplicar, y el parámetro axis determina si se aplica a las columnas (0) o a las filas (1)''')     

                        
                st.divider()
                st.write('##### Conversion de tipos de datos.')
                
                st.write('''**astype()** : convierte a otro tipo de dato (string, int64, float)''')         

                st.code('''df_diabetes['class_Bool'] = df_diabetes['class_Bool'].astype('int')''')
                df_diabetes['class_Bool'] = df_diabetes['class_Bool'].astype('int')
                st.code(df_diabetes.head(), language='html')


                st.write('Conversion de mas de una columna')
                
                st.code('''columnas_cod = ['class_Bool','class_texto']
df_diabetes[columnas_cod] = df_diabetes[columnas_cod].astype('string')''')
                
                
                st.write('##### Eliminar informacion duplicada.')

                st.write('''**drop()** : metodo que elimina filas (axis=0), o columnas (axis=1) de un DataFrame. Devuelve una copia modificada por defecto, 
aunque puede alterar el original con inplace=True''')
                
                data = ['class_texto', 'class_Bool']
                st.code('''data = ['class_texto', 'class_Bool']
df_diabetes.drop(data, axis=1, inplace=True)''')
                
                df_diabetes.drop(data, axis=1, inplace=True)
                st.code(df_diabetes.head(), language='html')        
                
                st.divider()
                
                st.write('##### Histogramas')   
        
                st.code('''import matplotlib.pyplot as plt
                
fig = plt.figure(figsize=(8,8))                
df_diabetes.hist(ax=fig.gca())                
plt.show()''')
        
        
                with st.container(width=1000):
                        fig = plt.figure(figsize=(8,8))
                        df_diabetes.hist(ax=fig.gca())
                        st.pyplot(fig)
                
                
                st.divider()

                st.code('''import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,3, figsize=(12,12))
sns.distplot(df_diabetes['preg'], ax=ax[0,0])
sns.distplot(df_diabetes['plas'], ax=ax[0,1])
sns.distplot(df_diabetes['pres'], ax=ax[0,2])
sns.distplot(df_diabetes['skin'], ax=ax[1,0])
sns.distplot(df_diabetes['test'], ax=ax[1,1])
sns.distplot(df_diabetes['mass'], ax=ax[1,2])
sns.distplot(df_diabetes['pedi'], ax=ax[2,0])
sns.distplot(df_diabetes['age'], ax=ax[2,1])''')
        
                with st.container(width=1000):
                        fig, ax = plt.subplots(3,3, figsize=(12,12))
                        sns.distplot(df_diabetes['preg'], ax=ax[0,0])
                        sns.distplot(df_diabetes['plas'], ax=ax[0,1])
                        sns.distplot(df_diabetes['pres'], ax=ax[0,2])
                        sns.distplot(df_diabetes['skin'], ax=ax[1,0])
                        sns.distplot(df_diabetes['test'], ax=ax[1,1])
                        sns.distplot(df_diabetes['mass'], ax=ax[1,2])
                        sns.distplot(df_diabetes['pedi'], ax=ax[2,0])
                        sns.distplot(df_diabetes['age'], ax=ax[2,1])        
                
                st.pyplot(fig)
                
                st.divider()    
                
                
                st.write('##### Diagrama de Densidad')
                
                st.code('''fig = plt.figure(figsize=(12,12))
df_diabetes.plot(ax=fig.gca(), kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()''')
                
                fig = plt.figure(figsize=(12,12))
                df_diabetes.plot(ax=fig.gca(), kind='density', subplots=True, layout=(3,3), sharex=False)

                st.pyplot(fig)
        
                st.code('''fig, ax = plt.subplots(3,3, figsize=(12,12))
sns.distplot(df_diabetes['preg'], hist=False, rug=True, ax=ax[0,0])
sns.distplot(df_diabetes['plas'], hist=False, rug=True, ax=ax[0,1])
sns.distplot(df_diabetes['pres'], hist=False, rug=True, ax=ax[0,2])
sns.distplot(df_diabetes['skin'], hist=False, rug=True, ax=ax[1,0])
sns.distplot(df_diabetes['test'], hist=False, rug=True, ax=ax[1,1])
sns.distplot(df_diabetes['mass'], hist=False, rug=True, ax=ax[1,2])
sns.distplot(df_diabetes['pedi'], hist=False, rug=True, ax=ax[2,0])
sns.distplot(df_diabetes['age'], hist=False, rug=True, ax=ax[2,1])''')
                
                fig, ax = plt.subplots(3,3, figsize=(12,12))
                sns.distplot(df_diabetes['preg'], hist=False, rug=True, ax=ax[0,0])
                sns.distplot(df_diabetes['plas'], hist=False, rug=True, ax=ax[0,1])
                sns.distplot(df_diabetes['pres'], hist=False, rug=True, ax=ax[0,2])
                sns.distplot(df_diabetes['skin'], hist=False, rug=True, ax=ax[1,0])
                sns.distplot(df_diabetes['test'], hist=False, rug=True, ax=ax[1,1])
                sns.distplot(df_diabetes['mass'], hist=False, rug=True, ax=ax[1,2])
                sns.distplot(df_diabetes['pedi'], hist=False, rug=True, ax=ax[2,0])
                sns.distplot(df_diabetes['age'], hist=False, rug=True, ax=ax[2,1]) 
                st.pyplot(fig)        
        
                st.divider()          
                
                st.write('##### Boxplot')

                st.code('''fig = plt.figure(figsize=(12,12))
df_diabetes.plot(ax=fig.gca(), kind='box', subplots=True, layout=(3,3), sharex=False)
plt.show()''')

                fig = plt.figure(figsize=(12,12))
                df_diabetes.plot(ax=fig.gca(), kind='box', subplots=True, layout=(3,3), sharex=False)
                st.pyplot(fig)
                
                st.divider()    
        
        
                st.code('''fig, ax = plt.subplots(3,3, figsize=(12,12))
sns.boxplot(df_diabetes['preg'], ax=ax[0,0])
sns.boxplot(df_diabetes['plas'], ax=ax[0,1])
sns.boxplot(df_diabetes['pres'], ax=ax[0,2])
sns.boxplot(df_diabetes['skin'], ax=ax[1,0])
sns.boxplot(df_diabetes['test'], ax=ax[1,1])
sns.boxplot(df_diabetes['mass'], ax=ax[1,2])
sns.boxplot(df_diabetes['pedi'], ax=ax[2,0])
sns.boxplot(df_diabetes['age'], ax=ax[2,1])''')
        
        
                fig, ax = plt.subplots(3,3, figsize=(12,12))
                sns.boxplot(df_diabetes['preg'], ax=ax[0,0])
                sns.boxplot(df_diabetes['plas'], ax=ax[0,1])
                sns.boxplot(df_diabetes['pres'], ax=ax[0,2])
                sns.boxplot(df_diabetes['skin'], ax=ax[1,0])
                sns.boxplot(df_diabetes['test'], ax=ax[1,1])
                sns.boxplot(df_diabetes['mass'], ax=ax[1,2])
                sns.boxplot(df_diabetes['pedi'], ax=ax[2,0])
                sns.boxplot(df_diabetes['age'], ax=ax[2,1])
                
                st.pyplot(fig)          
        
        
        
        
        
        
        
                st.write('##### Codificado get_dummies')
                
                st.write('''**get_dummies()** : se utiliza para convertir variables categóricas en variables ficticias o binarias (con valores de 0 o 1).       
        Este proceso, también conocido como codificación one-hot, es fundamental para preparar datos para algoritmos de aprendizaje automático que requieren entradas numéricas.        
        La función crea nuevas columnas para cada categoría única en la variable original, indicando la presencia (1) o ausencia (0) de esa categoría en cada fila      
        El parametro drop_first = True elimina la primera categoria para evitar multicolinealidad.''')
                

                st.divider()
                st.write('##### Escalamiento')                
                st.write('''La mayoria de los algorimos de Machine Learning funcionan mucho mejor si las caracteristicas estan en la misma escala.      
        Sin embargo, algunos como los basados en arboles de decision que no lo necesitan.''')
                
                st.write('''**Normalizacion:** Consiste en el re-escalado de las caracteristicas dentro de un rango [0...1], [min...max].      
        Se aplica la siguiente expresion: Xnorm = (X-Xmin)/(Xmax-Xmin)''')                

                st.write('''**Estandarizacion:** Requiere que se tome cada dato, se le reste el valor medio de esa caracteristica, y esa diferencia se divida por el desvio estandar.  
        La estandarizacion puede ser mas conveniente para modelos que usan algoritmos del descenso del gradiente, porque facilita la convergencia del mismo.''')
                

                st.divider()
                st.write('##### Valoraciones Cruzadas')
                
                st.write('''Es una tecnica utilizada para evaluar los resultados de un analisis estadistico y garantizar que son independientes de la particion entre datos de entrenamimento y prueba.         
Los resultados solo son significativos si los conjuntos se extraen de la misma poblacion.       
Consiste an ajustar y predecir usando el mismo modelo y promediar el rendimiento obtenido de las medidas de evaluacion sobre diferentes particiones provenientes del mismo conjunto de entrenamiento.   

Existen 2 metodos:   
        
**Con retencion**: Se separa el dataset en 2 subconjuntos: entrenamiento y prueba. Se ajusta el modelo con el conjunto de entrenamiento y se observa el rendimiento sobre el conjuto de testeo,   
repitiendo varias veces el mismo proceso hasta seleccionar los mejores parametros del modelo. Sin embargo, de esta manjera el conjunto de testeo acaba por ser parte del entrenamiento y el modelo puede sobreajustarse.             
Para evitar este efecto, conviene dividir el dataset en 3 subconjuntos: entrenamiento, prueba y validacion. El modelo se entrena con el conjunto de entrenamiento, se utiliza la validacion para obtener mejores parametros y solo se usa el conjunto de testeo para determinar el rendimiento del modelo.          
El inconveniente es que la estimacion del rendimiento puede ser muy sensible a como se divide el dataset.

**K iteraciones (k-fold CV)**: es una tecnica mas robusta y precisa por la manera de dividir el dataset. Se separa el dataset en conjunto de entrenamiento y conjunto de testeo.        
Se divide aleatoriamente el conjunto de entrenamiento en k subconjuntos sin reemplazo. De estos, k-1 se usan para entrenar el modelo y el restante para evaluar el rendimiento. 
Luego se calcula el rendimiento medio de los modelos a partir de las estimaciones independientes. Finalmente, elegido el modelo con los parametros que dan el mejor rendimiento,        
se lo vuele a entrenar con el conjunto de entrenamiento entero y se obtiene una estimacion final del modelo con el conjunto de testeo independiente.

Para conjuntos de datos pequeños, se puede usar un enfoque alternativo: LOOCV (validacion cruzada dejando uno afuera). Se toma k=n (nro de muestras) y en cada iteracion, se deja una sola sobre la cual se testea.     
Al final, todas las muestras fueron usadas para testear dentro del conjunto de entrenamiento.   
Para clases desbalanceadas, es mejor usar Validacion Cruzada Estratificada de iteraciones (k-fold stratified CV), ya que mejora la estimacion de sesgo y varianza.      
Se respetan las proporciones de clase en cada iteracion, lo que garantiza que siempre se respeten las proporciones del conjunto de entrenamiento.''') 
        
                
        if opcion_seleccionada == 'Machine Learning':
                st.write('##### Inteligencia Artificial')
                st.write('''Subdisciplina del campo de la informática, que busca la creación de máquinas que puedan imitar comportamientos inteligentes.    
        La inteligencia artificial puede ser Fuerte: sistemas que pueden realizar multitud de tareas, incluso con un alto nivel de complejidad.     
        Puede ser débil: sistemas que pueden cumplir con un conjunto limitado de tareas.
        
**Areas de aplicación:** Visión, Voz, Procesado de Lenguaje Natural (PLN), Sistemas Expertos, Robots, Machine Learning (Deep Learning)''')
                st.write('---')
                st.write('##### Machine Learning')
                st.write('''Se refiere a un amplio conjunto de técnicas informáticas que nos permiten dar a las computadoras la capacidad de aprender sin ser explícitamente programadas.''')

                st.write('##### Aprendizaje Supervisado')
                
                st.write('''Estos algoritmos aprenden a partir de casos previamentes etiquetados. Objetivo aprender a mapear las entradas en salidas,     
midiendo el error de lo aprendido con el dato real. Pueden ser de Clasificacion o Regresion.''')

                st.write('''**Clasificacion** :  El algoritmo intenta etiquetar cada ejemplo eligiendo entre dos o mas clases diferentes. Usan las caracteristicas aprendidas de los datos de capacitacion sobre datos nuevos,
no vistos previamente, para predecir sus etiquetas de clase. Elegir entre dos clases se denomina clasificacion binaria. Elegir entre mas de dos clases se denomina clasifiacion multiclase.''')

                st.write('''**Regresion** : Donde se predice un valor real basado en entradas pasadas. Estos algoritmos se usan para predecir valores de salida basados en algunas caracteristicas de entrada obtenidas de los datos.
Los valores de salida en este caso son continuos.''')

                st.write('##### Aprendizaje No supervisado')
                st.write('''Estos algoritmos no cuentan con un conocimiento previo. Se enfrentan al caos de datos con el objetivo de encontrar patrones que permitan organizarlos de alguna manera. 
Pueden ser de Clustering o Reduccion de dimensiones.            

**Aprendizaje por Refuerzo:** Ei sistema aprende a partir de su propia experiencia, en base a un proceso de prueba, error y recompensas si toma decisiones correctas.''')
                


def ml_aprendizaje_NoSupervisado():
    
    
    buffer = io.StringIO()   
    st.write('#### Definición') 

    st.write('''Es un tipo de aprendizaje automático que utiliza algoritmos para encontrar patrones en datos sin etiquetar, es decir, sin intrucciones o 'respuestas correctas' predifinidas.
Su objetivo es descubrir la estructura oculta en los datos, lo que permite agrupar datos similares (clustering), reducir la complejidad de los datos (reducción de dimensionalidad) o identificar anomalías.''')

    st.write('---')


    opciones_NoSupervisado = ['Resumen','Kmeans (Mall Customers)','Kmeans (Universities)', 'Agr. Jerarquico', 'Agr. por Densidad (DBSCAN)']
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_NoSupervisado)
        st.success(f'##### **{opcion_seleccionada}** ')
    

    if opcion_seleccionada == 'Resumen':
        st.write('#### Características principales')
        st.write('''* **Datos sin etiquetar**: Los algoritmos trabajan con datos que no tienen una variable de salida o etiqueta asociada.    
A diferencia del aprendizaje supervisado, no hay "respuestas correctas" proporcionadas durante el entrenamiento.''') 
        st.write('''* **Descubrimiento de patrones**: Los modelos identifican automáticamente similitudes y diferencias dentro de los datos para encontrar patrones y estructuras subyacentes.''') 
        st.write('''* **Autonomía**:  El algoritmo opera de forma independiente para descubrir la estructura de los datos sin necesidad de intervención humana o guía explícita sobre el resultado esperado.''') 

        st.write('---')
        st.write('#### Aplicaciones comunes')
        st.write('''* **Agrupación de clientes**: Segmentar clientes en grupos basados en sus comportamientos de compra para personalizar ofertas, como se hace en el comercio electrónico.''')   
        st.write('''* **Sistemas de recomendación**: Descubrir tendencias en datos históricos para sugerir productos o contenido complementario.''') 
        st.write('''* **Detección de anomalías**: Identificar puntos de datos inusuales que podrían indicar errores, fallos de equipos o infracciones de seguridad.''') 
        st.write('''* **Procesamiento de imágenes médicas**: Ayudar en tareas de radiología y patología para analizar y clasificar imágenes de forma más rápida y precisa.''') 
    
        st.write('---')    
        st.write('#### Técnicas comunes')    
        st.write('''* **Agrupación (clustering)**: Agrupar puntos de datos en clústeres basándose en su similitud. Ejemplos son los algoritmos K-Means.''')    
        st.write('''* **Reducción de dimensionalidad**: Reducir el número de variables en un conjunto de datos mientras se conserva la información importante.  
    Un ejemplo es el Análisis de Componentes Principales (PCA).''')    


    if opcion_seleccionada == 'Agr. por Densidad (DBSCAN)':

        st.write('##### Data Frame con dato de clientes de una tienda')         
    
        st.code('''from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler''')
        
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import MinMaxScaler
        

        st.code('''df_arboles = pd.read_csv('DataFrames/arbolado-en-espacios-verdes.csv)
st.dataframe(df_arboles.head(10)) ''')
        
        df_arboles = pd.read_csv('DataFrames/arbolado-en-espacios-verdes.csv')
        st.dataframe(df_arboles.head(10))
        
         
        st.code('''# info
df_arboles.info()''') 
        
        df_arboles.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          

        st.write('---')
        st.write('**Cantidad de especies diferentes**')

        st.code('df_arboles[\'id_especie\'].nunique()')
        
        especies = df_arboles['id_especie'].nunique()
        st.code(especies, language='html')

        st.write('**Columnas que interesan**')
        
        st.code('df_data = df_arboles[[\'diametro\',\'altura_tot\',\'nombre_com\']]')

        df_data = df_arboles[['diametro','altura_tot','nombre_com']]
        
        st.write('**Modificar nombre de altura_tot -> altura**')
        
        st.code('df_data.rename(columns={\'altura_tot\':\'altura\'}, inplace=True)')
        df_data.rename(columns={'altura_tot':'altura'}, inplace=True)

        st.dataframe(df_data.head(10))

        st.write('**Filtrado por especies**')

        st.code('''especies = (df_data['nombre_com'] == 'Jacarandá') | (df_data['nombre_com'] == 'Palo borracho rosado') | (df_data['nombre_com'] == 'Eucalipto') 
| (df_data['nombre_com'] == 'Ceibo') 
df_data_especies = df_data[especies]''')     
        
        st.code('''# Otra forma de filtro
especies = df_data['nombre_com'].isin(['Jacarandá','Palo borracho rosado','Eucalipto','Ceibo'])''')
        
        especies = df_data['nombre_com'].isin(['Jacarandá','Palo borracho rosado','Eucalipto','Ceibo'])             
#       especies = (df_data['nombre_com'] == 'Jacarandá') | (df_data['nombre_com'] == 'Palo borracho rosado') | (df_data['nombre_com'] == 'Eucalipto') | (df_data['nombre_com'] == 'Ceibo') 
        
        df_data_especies = df_data[especies]
        st.dataframe(df_data_especies.head(10))

        st.write('**Seleccion de una especie**: Eucalipto')

        st.write('Filtrar aquellos arboles con diametro > 10cm par sacar valores erroneos.')
        st.code('''eucaliptus = (df_data_especies['nombre_com'] == 'Eucalipto') & (df_data_especies['diametro'] > 10)
df_eucalipto = df_data_especies[eucaliptus]
df_eucalipto.reset_index(inplace=True)      # resetear indice
df_eucalipto.drop(['index','nombre_com'], axis=1, inplace=True)  # eliminar indice y nombre''')
        
        eucaliptus = (df_data_especies['nombre_com'] == 'Eucalipto') & (df_data_especies['diametro'] > 10)

        df_eucalipto = df_data_especies[eucaliptus]
        df_eucalipto.reset_index(inplace=True)      # resetear indice
        df_eucalipto.drop(['index','nombre_com'], axis=1, inplace=True)  # eliminar indice y nombre
         
        st.dataframe(df_eucalipto.head(10))

        st.write('---')
        
        st.code('''fig,ax = plt.subplots()
ax.plot(df_eucalipto['diametro'],df_eucalipto['altura'], color='blue', linestyle='none', marker='o', markersize =3, alpha=.3)                    
plt.title('Grafico de Eucaliptus (diametro vs altura)', size=10)
plt.xlabel('Diametro (cm)', size=8)
plt.ylabel('Altura (m)', size=8)                    
                    
st.pyplot(fig)''')
            
        
        
        with st.container(width=800):

            fig,ax = plt.subplots()
            ax.plot(df_eucalipto['diametro'],df_eucalipto['altura'], color='blue', linestyle='none', marker='o', markersize =3, alpha=.3)


            plt.title('Grafico de Eucaliptus (diametro vs altura)', size=10)
            plt.xlabel('Diametro (cm)', size=8)
            plt.ylabel('Altura (m)', size=8)

            st.pyplot(fig)

        st.write('---')
        st.write('##### Normalización de los datos')
        
        st.write('''* **MinMaxScaler**: es una tecnica de preprocesamiento que transforma las caracteristicas escalonandolas a un rango fijo, 
    generalmente entre 0 y 1, ajustando los valores proporcionalmente para que el minimo se convierta en 0 y el maximo en 1, manteniendo la forma original de la distribucion de los datos.''')
        
        st.code('''scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_eucalipto)                

df_eucalipto_escaled = pd.DataFrame(df_scaled, columns=df_eucalipto.columns)''')
        
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_eucalipto)
        
        df_eucalipto_escaled = pd.DataFrame(df_scaled, columns=df_eucalipto.columns)
        st.write(df_eucalipto_escaled.head(10))

        with st.container(width=800):

            fig,ax = plt.subplots()
            ax.plot(df_eucalipto_escaled['diametro'],df_eucalipto_escaled['altura'], color='blue', linestyle='none', marker='o', markersize =3, alpha=.3)


            plt.title('Grafico de Eucaliptus (escalado)', size=10)
            plt.xlabel('Diametro', size=8)
            plt.ylabel('Altura', size=8)

            st.pyplot(fig)

        st.write('---')
        st.write('**Parametrizacion de DBSCAN usaando el método de la rodilla**')
        
        st.code('''estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(df_eucalipto_escaled)                
dist = metrics.DistanceMetric.get_metric('euclidean')                
matsim = dist.pairwise(X_pca)   
minPts = 5   
A = kneighbors_graph(X_pca, minPts, include_self=False)   
Ar = A.toarray()   
seq = [] 
  
for i,s in enumerate(X_pca):
    for j in range(len(X_pca)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])   
   
seq.sort() 
 
fig,ax = plt.subplots()   
ax.plot(seq)
st.pyplot(fig)''')
        
        
        estimator = PCA(n_components=2)
        X_pca = estimator.fit_transform(df_eucalipto_escaled)
        dist = metrics.DistanceMetric.get_metric('euclidean')
        matsim = dist.pairwise(X_pca)
        minPts = 5
        A = kneighbors_graph(X_pca, minPts, include_self=False)
        Ar = A.toarray()
        seq = []
        for i,s in enumerate(X_pca):
            for j in range(len(X_pca)):
                if Ar[i][j] != 0:
                    seq.append(matsim[i][j])

        seq.sort()
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            ax.plot(seq)
            
            st.pyplot(fig)

        st.write('---')
        st.write('##### Ejecución DBSCAN')
        
        st.code('''dbscan = DBSCAN(eps=0.032, min_samples=5, metric='euclidean').fit(df_eucalipto_escaled)
clusters = dbscan.fit_predict(df_eucalipto_escaled)''')
        
        dbscan = DBSCAN(eps=0.032, min_samples=10, metric='euclidean').fit(df_eucalipto_escaled)
        clusters = dbscan.fit_predict(df_eucalipto_escaled)
        #df_values = df_eucalipto.values
        
        st.code('''fig,ax = plt.subplots()
plt.scatter(df_eucalipto['diametro'], df_eucalipto['altura'],c=clusters,cmap='viridis')
plt.xlabel('Diametro (cm)') 
plt.ylabel('Altura (m)')   

st.pyplot(fig)''')
        
        # Graficacion de clusters
        with st.container(width=800):            
            fig,ax = plt.subplots()
            plt.scatter(df_eucalipto['diametro'], df_eucalipto['altura'],c=clusters,cmap='viridis')
            #plt.scatter(df_values[:,0], df_values[:,1],c=clusters,cmap='viridis')
            plt.xlabel('Diametro (cm)')
            plt.ylabel('Altura (m)')

            st.pyplot(fig)
            
        st.write('---')
        st.write('Clusters')
        st.code('np.unique(clusters)')
        st.code(np.unique(clusters))    

        st.code('''df_clusters = pd.DataFrame()
df_clusters['altura'] = df_eucalipto['altura'].values                
df_clusters['diametro'] = df_eucalipto['diametro'].values 
df_clusters['label'] = clusters''')
        
        df_clusters = pd.DataFrame()
        df_clusters['altura'] = df_eucalipto['altura'].values
        df_clusters['diametro'] = df_eucalipto['diametro'].values
        df_clusters['label'] = clusters
        
      
        
        st.write('**Cantidad por grupo**')
        
        st.code('''df_cantidad_grupo = pd.DataFrame()
df_cantidad_grupo['cantidad'] = df_clusters.groupby('label').size()''')
        
        df_cantidad_grupo = pd.DataFrame()
        df_cantidad_grupo['cantidad'] = df_clusters.groupby('label').size()
        
        st.dataframe(df_cantidad_grupo, width=200)        

        st.write('Eliminar puntos de ruido (-1)')
        st.code('''filtro = df_clusters['label'] > -1
df_clusters = df_clusters[filtro]''')
        
        filtro = df_clusters['label'] > -1
        df_clusters = df_clusters[filtro]

        st.write('**Grafico del DataFrame limpio**')
        
        st.code('''fig,ax = plt.subplots()
ax.scatter(x=df_clusters['diametro'], y=df_clusters['altura'], color='#0004ff')  
plt.xlabel('Diametro (cm)')
plt.ylabel('Altura (m)')
plt.title('Arboles sin outliers')              
st.pyplot(fig)''')
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            ax.scatter(x=df_clusters['diametro'], y=df_clusters['altura'], color="#0004ff")

            plt.xlabel('Diametro (cm)')
            plt.ylabel('Altura (m)')
            plt.title('Arboles sin outliers')
            st.pyplot(fig)



    if opcion_seleccionada == 'Agr. Jerarquico':
    
        st.write('##### Data Frame con dato de clientes de una tienda')         
    
        st.code('''import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering ''')
        
        
        st.code('''df_mall = pd.read_csv('DataFrames/Mall_customers.csv', index_col='CustomerID')
st.dataframe(df_mall.head(10)) ''')
        
        df_mall = pd.read_csv('DataFrames/Mall_customers.csv', index_col='CustomerID')
        st.dataframe(df_mall.head(10))
         
        st.code('''# info
df_mall.info()''') 
        
        df_mall.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          

        st.write('---')
        st.write('##### Eliminar columna género')
        st.write('Se elimina la columna género ya que es irrelevante.')
        
        st.code('df_mall.drop(\'Gender\', axis=1, inplace=True)')
        df_mall.drop('Gender', axis=1, inplace=True)


        st.write('---')
        st.write('##### Generación del Dendograma')

        st.code('''dend = hc.dendrogram(hc.linkage(df_mall, method='ward'))

fig,ax = plt.subplots()

plt.title('Dendograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia euclidiana')
plt.savefig('dendograma.png', dpi=1200)
        
st.pyplot(fig)''')
        
        dend = hc.dendrogram(hc.linkage(df_mall, method='ward'))

       # with st.container(width=600):
        #    fig,ax = plt.subplots()

         #   plt.title('Dendograma')
          #  plt.xlabel('Clientes')
           # plt.ylabel('Distancia euclidiana')
           # plt.savefig('dendograma.png', dpi=1200)
            
           # st.pyplot(fig)

        imagen = Image.open('Imagenes/dendrograma1.png')
        st.image(imagen, width=800)

        st.code('''clust = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
plo = clust.fit_predict(df_mall)''')
        
        clust = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
        plo = clust.fit_predict(df_mall)

        st.write('Generar el Dendrograma para observar donde corta según n_clusters=6')

        st.code('''Z = hc.linkage(df_mall, method='ward')
dend = hc.dendrogram(Z)   
             
fig,ax = plt.subplots()
plt.title('Dendograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia euclidiana')           
''')


        Z = hc.linkage(df_mall, method='ward')
        dend = hc.dendrogram(Z)
        
        fig,ax = plt.subplots()
        plt.title('Dendograma')
        plt.xlabel('Clientes')
        plt.ylabel('Distancia euclidiana')

        st.write('Calcular altura para 6 clusters')
        
        st.code('''n_cluster = 6
altura_corte = Z[-(n_cluster),2]    # altura exacta
plt.axhline(y=altura_corte, color='r', linestyle='--', label=f'corte: {altura_corte:.2f}')
plt.legend()

st.pyplot(fig)''')
        
        n_cluster = 6
        altura_corte = Z[-(n_cluster),2]    # altura exacta
        plt.axhline(y=altura_corte, color='r', linestyle='--', label=f'corte: {altura_corte:.2f}')
        plt.legend()

       # st.pyplot(fig)
        imagen = Image.open('Imagenes/dendograma2.png')
        st.image(imagen, width=800)

        st.write('---')
        st.write('**Clientes en cada cluster**')
        
     
        st.code('''unique_elements, counts = np.unique(plo, return_counts=True)
resultado = zip(unique_elements, counts)                

for key, value in resultado:    
    st.write(f'{key} : {value}')            
''')
     
        unique_elements, counts = np.unique(plo, return_counts=True)
        resultado = zip(unique_elements, counts)
            
        for key, value in resultado:
            st.write(f'{key} : {value}')


        st.write('##### Grafico Plotly para la clasificacion (3d)')
        
        st.code('import plotly.express as px')
        
        import plotly.express as px

        st.code('''fig = px.scatter_3d(data_frame=df_mall, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)', width=1000, height=800, color=preds) # clusters
ig.add_trace(px.scatter_3d(data_frame=centros, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)').update_traces(                
    marker=dict(size=4, symbol='x', color='black')).data[0]) # centroides   
    
st.plotly_chart(fig)''')
        
        fig = px.scatter_3d(data_frame=df_mall, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)', width=1000, height=800, color=plo) # clusters
       # fig.add_trace(px.scatter_3d(data_frame=centros, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)').update_traces(
        #    marker=dict(size=4, symbol='x', color='black')).data[0]) # centroides
        
        st.plotly_chart(fig)


    if opcion_seleccionada == 'Kmeans (Universities)':
        
        st.write('##### Data Frame con datos de Universidades') 
        st.code('''from sklearn.cluster import KMeans''')

        st.code('''df_college = pd.read_csv('DataFrames/College_Data', index_col=0)
df_college.head(10)            
                ''')

        df_college = pd.read_csv('DataFrames/College_Data', index_col=0)
        st.dataframe(df_college.head(10))


        st.code('''# info
df_college.info()''') 
        
        df_college.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')  

        st.write('---')
        

        col_1, col_2 = st.columns(2)
        with col_1:
            st.write('##### Scatterplot (Room.Board vs Grad.Rate)')
            fig = sns.lmplot(
                data=df_college, 
                x='Room.Board',
                y='Grad.Rate',
                hue='Private',
                fit_reg=False,
            )

            st.pyplot(fig) 
        

        with col_2:
            st.write('##### Scatterplot (Outstate vs F.Undergrad)')
            fig = sns.lmplot(
                data=df_college, 
                x='Outstate',
                y='F.Undergrad',
                hue='Private',
                fit_reg=False,
            )

            st.pyplot(fig)  

        st.write('---')
        st.write('##### Histogram Outstate')
        
        with st.container(width=800):
        
            fig,ax = plt.subplots()
            sns.histplot(
                data=df_college,
                x='Outstate', 
                kde=False, 
                hue='Private',
                alpha=0.5,
                bins=20
            )

            st.pyplot(fig) 


        st.write('---')
        st.write('##### Histogram Grad.Rate')
        
        with st.container(width=800):
        
            fig,ax = plt.subplots()
            sns.histplot(
                data=df_college,
                x='Grad.Rate', 
                kde=False, 
                hue='Private',
                alpha=0.5,
                bins=20
            )

            st.pyplot(fig) 


        st.write('---')
        st.write('**Crear Clusters de K Means**')
        
        st.code('''from sklearn.cluster import KMeans''')
        
        st.write('**Crear instancia de K Means con 2 clusters**')
        st.code('kmeans = KMeans(n_clusters=2)')
        kmeans = KMeans(n_clusters=2)
        
        st.write('**Entrenamiento del modelo**')
        st.code('''df_college_fit = df_college.drop('Private',axis=1)
kmeans.fit(df_college_fit)''')
        
        df_college_fit = df_college.drop('Private',axis=1)
        kmeans.fit(df_college_fit)
        
        st.code(kmeans.cluster_centers_, language='html')  
        st.write('**Ubicación de los centroides**')    
        centroides = kmeans.cluster_centers_



    if opcion_seleccionada == 'Kmeans (Mall Customers)':
    
        st.write('##### Data Frame con dato de clientes de una tienda')         
        st.code('''from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score ''')
        
        st.code('''df_mall = pd.read_csv('DataFrames/Mall_customers.csv', index_col='CustomerID')
st.dataframe(df_mall.head(10)) ''')
        
        df_mall = pd.read_csv('DataFrames/Mall_customers.csv', index_col='CustomerID')
        st.dataframe(df_mall.head(10))
         
        st.code('''# info
df_mall.info()''') 
        
        df_mall.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          

        st.write('---')
        st.write('##### Eliminar columna género')
        st.write('Se elimina la columna género ya que es irrelevante.')
        
        st.code('df_mall.drop(\'Gender\', axis=1, inplace=True)')
        df_mall.drop('Gender', axis=1, inplace=True)


        st.write('---')
        st.write('##### Histograma') 
        
        col1,col2,col3 = st.columns(3)
        
        with col1:
            fig,ax = plt.subplots()   
            ax.hist(x=df_mall['Age'], bins=15, edgecolor='#000000', color='#8b92cc', alpha=.8)
            ax.set_title('Histograma (Age)')

            st.pyplot(fig)
        with col2:
            fig,ax = plt.subplots()   
            ax.hist(x=df_mall['Annual Income (k$)'], bins=15, edgecolor='#000000', color='#8b92cc', alpha=.8)
            ax.set_title('Histograma (Annual Income k$)')

            st.pyplot(fig)
        with col3:
            fig,ax = plt.subplots()   
            ax.hist(x=df_mall['Spending Score (1-100)'], bins=15, edgecolor='#000000', color='#8b92cc', alpha=.8)
            ax.set_title('Spending Score (1-100)')

            st.pyplot(fig)

        st.write('---')
        st.write('##### Parámetros skmeans')
        st.write('* **n_clusters**: El número de clústers a formar, así como el número de centroides a generar (por defecto 8).') 
        st.write('''* **init**: k-means++ selecciona los centroides iniciales del conglomerado mediante un muestreo basado en una distribución de probabilidad empírica
de la contribución de los puntos a la inercia general.      
random elige n_clusters observaciones (filas) al azar a partir de los datos para los centroides iniciales.
Si se pasa una matriz, debe tener forma (n_clusters, n_features) y proporcionar los centros iniciales.''')  

        st.write('''* **n_init**: número de veces que se ejecuta el algoritmo k-means con diferentes valores de semilla de centroide.       
El resultado final es el mejor resultado de n_init ejecuciones consecutivas en términos de inercia (por defecto auto).''')
        
        st.write('* **max_iter**: número máximo de iteraciones del algoritmo k-means para una sola ejecución (por defecto 300).')
        st.write('* **tol**: tolerancia relativa con respecto a la norma de Froebenius de la diferencia de los centros de los grupos de dos iteraciones consecutivas para declarar la convergencia.(por defecto 1e-4)')
        st.write('''* **algotithm**: algoritmo k-means a utilizar. El algoritmo clásico es lloyd, puede usarse también elkan.''')


        st.write('---')
        st.write('##### Elección k usando el método de silueta (el máximo)')

        st.code('''wcss = []
calinski = []
davies = []                

for n_cluster in range(2,11):
    km = KMeans(n_clusters=n_cluster, init='k-means++', random_state=16)
    preds = km.fit_predict(df_mall)
    sil_coeff = silhouette_score(df_mall, preds, metric='euclidean')
    st.write(f'Para n_clusters={n_cluster}, el coeficiente de Silhouette es {sil_coeff}')
    wcss.append(km.inertia_)   # inertia_ -> distancia intra clusters
    calinski.append(calinski_harabasz_score(df_mall, preds))
    davies.append(davies_bouldin_score(df_mall, preds))
    sample_silhouette_values = silhouette_samples(df_mall, preds)

    fig = plt.figure()
    y_lower = 10
        for i in range(n_cluster):
            ith_cluster_silhouette_values = sample_silhouette_values[preds == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
                    
            color = cm.nipy_spectral(float(i)/n_cluster)
            plt.fill_betweenx(
                 np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=.7
            )
            
            plt.text(-0.05, y_lower+0.5*size_cluster_i, str(i))
            y_lower = y_upper + 10
                    
        plt.title(f'Grafico de silueta para los clusters, k = {n_cluster}')
        plt.xlabel('Valores de coeficientes de Silueta')
        plt.ylabel('Etiqueta cluster')

        plt.axvline(x=sil_coeff, color='red', linestyle='--')
                
        plt.yticks([])
        plt.xticks([-0.1,0.2,0.4,0.6,0.8,1])

st.pyplot(fig)
''')
        
        
        
        wcss = []
        calinski = []
        davies = []
        
        st.write('##### Grafico del metodo de la siuleta')
        
        with st.container(width=800):
        
            for n_cluster in range(2,11):
                km = KMeans(n_clusters=n_cluster, init='k-means++', random_state=16)
                preds = km.fit_predict(df_mall)
                sil_coeff = silhouette_score(df_mall, preds, metric='euclidean')
                st.write(f'Para n_clusters={n_cluster}, el coeficiente de Silhouette es {sil_coeff}')
                wcss.append(km.inertia_)   # inertia_ -> distancia intra clusters
                calinski.append(calinski_harabasz_score(df_mall, preds))
                davies.append(davies_bouldin_score(df_mall, preds))
                sample_silhouette_values = silhouette_samples(df_mall, preds)

                fig = plt.figure()
                y_lower = 10
                for i in range(n_cluster):
                    ith_cluster_silhouette_values = sample_silhouette_values[preds == i]
                    ith_cluster_silhouette_values.sort()
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    color = cm.nipy_spectral(float(i)/n_cluster)
                    plt.fill_betweenx(
                        np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=.7
                    )
            
                    plt.text(-0.05, y_lower+0.5*size_cluster_i, str(i))
                    y_lower = y_upper + 10
                    
                plt.title(f'Grafico de silueta para los clusters, k = {n_cluster}')
                plt.xlabel('Valores de coeficientes de Silueta')
                plt.ylabel('Etiqueta cluster')

                plt.axvline(x=sil_coeff, color='red', linestyle='--')
                
                plt.yticks([])
                plt.xticks([-0.1,0.2,0.4,0.6,0.8,1])

                st.pyplot(fig)


        
        st.write('---')
        st.write('##### Grafico del metodo del codo')
        
        st.code('''fig,ax = plt.subplots()
ax.plot(range(2,11),wcss,label='X Square', color='blue', linewidth=2)

plt.title('Grafico Metodo del codo')
plt.xlabel("Numero de clusters")
plt.ylabel("WCC")

st.pyplot(fig)               
                
''')
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            ax.plot(range(2,11),wcss,label='X Square', color='blue', linewidth=2)

            plt.title('Grafico Metodo del codo')
            plt.xlabel("Numero de clusters")
            plt.ylabel("WCC")

            st.pyplot(fig)
        
        st.write('---')
        st.write('##### Grafico de Calinski-Harabasz')
        
        st.code('''fig,ax = plt.subplots()
ax.plot(range(2,11),calinski,label='X Square', color='red', linewidth=2)                
                
plt.title('Grafico Calinski-Harabasz')    
plt.xlabel("Numero de clusters")
plt.ylabel("C-H score")   

st.pyplot(fig)        
''')
        
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            ax.plot(range(2,11),calinski,label='X Square', color='red', linewidth=2)

            plt.title('Grafico Calinski-Harabasz')
            plt.xlabel("Numero de clusters")
            plt.ylabel("C-H score")

            st.pyplot(fig)


        st.write('---')
        st.write('##### Grafico de Davies-Boulder')
        
        st.code('''fig,ax = plt.subplots()
ax.plot(range(2,11),davies,label='X Square', color='green', linewidth=2)                
                
plt.title('Grafico Davies-Boulder')    
plt.xlabel("Numero de clusters")
plt.ylabel("D-B score")   

st.pyplot(fig)        
''')
        
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            ax.plot(range(2,11),davies,label='X Square', color='green', linewidth=2)

            plt.title('Grafico Davies-Boulder')
            plt.xlabel("Numero de clusters")
            plt.ylabel("D-B score")

            st.pyplot(fig)


        st.write('---')
        st.write('##### Planteo del modelo con 6 clusters y se clasifica')
        
        st.code('''km = KMeans(n_clusters=6, init='k-means++', n_init=100, max_iter=1000)
preds = km.fit_predict(df_mall)''')
        
        km = KMeans(n_clusters=6, init='k-means++', n_init=100, max_iter=1000)
        preds = km.fit_predict(df_mall)
        
        st.code('print(preds)')
        st.code(preds, language='html')
        
        st.write('**Posición de los centroides**')
        st.code('km.cluster_centers')
        st.code(km.cluster_centers_, language='html')
        
        st.code('pd.DataFrame(km.cluster_centers_, columns=[\'Age\',\'Annual Income (k$)\',\'Spending Score (1-1000)])')
        centros = pd.DataFrame(km.cluster_centers_, columns=['Age','Annual Income (k$)','Spending Score (1-100)'])
        centros

        st.write('##### Grafico Plotly para la clasificacion (3d)')
        
        st.code('import plotly.express as px')
        
        import plotly.express as px

        st.code('''fig = px.scatter_3d(data_frame=df_mall, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)', width=1000, height=800, color=preds) # clusters
ig.add_trace(px.scatter_3d(data_frame=centros, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)').update_traces(                
    marker=dict(size=4, symbol='x', color='black')).data[0]) # centroides   
    
st.plotly_chart(fig)''')
        
        fig = px.scatter_3d(data_frame=df_mall, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)', width=1000, height=800, color=preds) # clusters
        fig.add_trace(px.scatter_3d(data_frame=centros, x="Age", y="Annual Income (k$)", z= 'Spending Score (1-100)').update_traces(
            marker=dict(size=4, symbol='x', color='black')).data[0]) # centroides
        
        st.plotly_chart(fig)
        
        st.write('---')
        st.write('**Agregar las etiquetas de cada cluster al dataset**')
        
        st.code('''df_mall['preds'] = preds
st.dataframe(df_mall.head(10))''')
        
        df_mall['preds'] = preds
        st.dataframe(df_mall.head(10))
        
        st.write('---')
        st.write('**Clientes en cada cluster**')
        
        with st.container(width=200):
            st.write(df_mall['preds'].value_counts())
    
    
  
def ml_NLP():  
    buffer = io.StringIO()   
 
    st.write('#### Definición')
    st.write('''Los sistemas de recomendación son algoritmos que intentan 'predecir' los siguientes items (productos, canciones, etc.) que quarrá adquirir un usuario en particular.    
Los sistemas de recomendacion intentan personalizar al máximo lo que ofrecerán a cada usuario.''')    
    
    st.write('---')     
    
    
def ml_SisRecomendacion():    
    buffer = io.StringIO()      
   
    st.write('#### Definición')
    st.write('''Los sistemas de recomendación son algoritmos que intentan 'predecir' los siguientes items (productos, canciones, etc.) que quarrá adquirir un usuario en particular.    
Los sistemas de recomendacion intentan personalizar al máximo lo que ofrecerán a cada usuario.''')    
    
    st.write('---')        
   
    opciones_rs = ['Movies']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_rs)
        st.success(f'##### **{opcion_seleccionada}** ')

      
    if opcion_seleccionada == 'Movies':
        st.write('##### Data Frame con los datos de Peliculas') 
         

        st.code('''column_names = ['user_id','item_id','rating','timestamp']
df_movies = pd.read_csv('DataFrames/u.data', sep='t', names=column_names)                
df_movies.head(10)''')
        
        column_names = ['user_id','item_id','rating','timestamp']
        
        df_movies = pd.read_csv('DataFrames/u.data', sep='\t', names=column_names)
        st.dataframe(df_movies.head(10))

        
        st.code('''# info
df_movies.info()''') 
        
        df_movies.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')   
        
        st.write('---')
        st.write('**Titulos**')
        
        st.code('''movie_titles = pd.read_csv('DataFrames/Movie_Id_Titles')
movie_titles.head(10)''')
        
        movie_titles = pd.read_csv('DataFrames/Movie_Id_Titles')
        st.dataframe(movie_titles.head(10))  
         
        
        # Fusionar ambos dataframes
        st.code('''# Fusionar dataframes
df_movies_fus = pd.merge(df_movies, movie_titles)                
df_movies_fus.head(10)''')
        
        df_movies_fus = pd.merge(df_movies, movie_titles, on='item_id')
        st.dataframe(df_movies_fus.head(10)) 
         
        st.write('---')
        
        # Agrupar por titulo y rating
        st.code('''# Peliculas ordenadas por rating
df_ratings = pd.DataFrame(df_movies_fus.groupby('title')['rating'].mean()) 
df_ratings_ord = df_movies_fus.groupby('title')['rating'].mean().sort_values(ascending=False)        
df_ratings_ord.head(10)''')
        
        df_ratings = pd.DataFrame(df_movies_fus.groupby('title')['rating'].mean())
        
        df_ratings_ord = df_movies_fus.groupby('title')['rating'].mean().sort_values(ascending=False)
        
        with st.container(width=500):
            st.dataframe(df_ratings_ord.head(10)) 
         
  
        st.code('''# Peliculas con mayor audiencia
df_agrupado = df_movies_fus.groupby('title')['item_id'].count().sort_values(ascending=False)                
df_agrupado.head(10)''')
        
        df_agrupado = df_movies_fus.groupby('title')['item_id'].count().sort_values(ascending=False)
        
        with st.container(width=500):
            st.dataframe(df_agrupado.head(10))   
    
    
        st.write('---')

        st.code('''df_ratings['num of ratings'] = pd.DataFrame(df_movies_fus.groupby('title')['item_id'].count())
df_ratings.head(10)''')
        
        df_ratings['num of ratings'] = pd.DataFrame(df_movies_fus.groupby('title')['item_id'].count())
        st.write(df_ratings.head(10))
        st.write('---')
        
        st.write('**Histograma rating**')

        st.code('''sns.set_style('white')
                
fig,ax = plt.subplots()                
    data=df_ratings,
    x='rating', 
    kde=True, 
    color='#e9b33b', 
    edgecolor='#E8BE58',
    bins=70,
    alpha=0.5
)

plt.title('Histograma rating')  
st.pyplot(fig)              
''')
    

        sns.set_style('white')
        
        with st.container(width=800):
        
            fig,ax = plt.subplots()
            sns.histplot(
                data=df_ratings,
                x='rating', 
                kde=True, 
                color='#e9b33b', 
                edgecolor='#E8BE58',
                bins=70,
                alpha=0.5
            )
            plt.title('Histograma rating') 

            st.pyplot(fig)

        st.write('---')
        st.write('**Jointplot rating vs num of ratings**') 
        
        st.code('''sns.set_style('white')
graf = sns.jointplot(data=df_ratings, x='rating',y='num of ratings', kind='reg')
st.pyplot(graf)''')
        
        sns.set_style('white')
        
        with st.container(width=800):
            graf = sns.jointplot(data=df_ratings, x='rating',y='num of ratings', kind='reg')
            st.pyplot(graf)
 
 
        st.write('---')
        
        st.code('''moviemat = df_movies_fus.pivot_table(index='user_id', columns='title',values='rating')
moviemat.head(10)''')
        
        moviemat = df_movies_fus.pivot_table(index='user_id', columns='title',values='rating')
        st.dataframe(moviemat.head(10))
    
 
 
 
 
 
 
   
def ml_Bayes(): 
    buffer = io.StringIO()      
   
    st.write('#### Definición')
    st.write('''Es un algoritmo de clasificación probabilístico simple pero potente en aprendizaje automático que usa el Teorema de Bayes con una suposición "ingenua": que todas las características son independientes entre sí
dada la clase. A pesar de este supuesto irreal, funciona muy bien en la práctica, especialmente en clasificación de texto y análisis de sentimientos,             
calculando eficientemente probabilidades condicionales para predecir la clase de un nuevo dato.''')


    st.write('---')        
   
    opciones_bayes = ['Mensajes SMS','Vinos','Noticias']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_bayes)
        st.success(f'##### **{opcion_seleccionada}** ')



    if opcion_seleccionada == 'Noticias':
        st.write('##### Data Frame con los datos Noticias')  
        st.write('El algoritmo trata de clasificar a una noticia.')

        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.naive_bayes import GaussianNB as GaussianNB_sk
        from sklearn.naive_bayes import BernoulliNB as BernoulliNB_sk
        from sklearn.naive_bayes import CategoricalNB as CategorialNB_sk
        from sklearn.naive_bayes import MultinomialNB as MultinomialNB_sk
        from sklearn.naive_bayes import ComplementNB as ComplementNB_sk
        
        
        st.code('''# load dataset
df_news = pd.read_csv('DataFrames/uci-news-aggregator.csv')  
df_news.head(10)''')
        
        @st.cache_data
        def cargar_data_news():
            df_news = pd.read_csv('DataFrames/uci-news-aggregator.csv')
            
            return df_news


        df_news = cargar_data_news()
        st.dataframe(df_news.head(10))
        

 
        st.code('''# info
df_news.info()''') 
        
        df_news.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')  
 
        st.write('---')
        st.write('**Recategorizar valores de Categoria a numerico**')
        
        st.code('''# business, technology, entertainement, medicin
data = {'b':1,'t':2,'e':3,'m':4}    
df_news['CATEGORY'] = df_news['CATEGORY'].map(data)
df_news.head(10)                ''')
        
        
        # business, technology, entertainement, medicin
        data = {'b':1,'t':2,'e':3,'m':4}
        
        df_news['CATEGORY'] = df_news['CATEGORY'].map(data)
        st.dataframe(df_news.head(10))        
        
        # Distribuciones de las clases
        st.code('''# Distribuciones de las clases
valores = df_news['CATEGORY'].value_counts()''')
        valores = df_news['CATEGORY'].value_counts()
        st.code(f'{valores}', language='html')

        st.write('---')
        
        st.write('##### Separación de los datos del modelo')

        st.code('''X = df_news['TITLE']
y = df_news['CATEGORY'] 
le = LabelEncoder() 
y = le.fit_transform(Y)              

X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)''')

        X = df_news['TITLE']
        Y = df_news['CATEGORY']
        le = LabelEncoder()
        y = le.fit_transform(Y)
        

        X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

        st.write('---')
        
        
        st.code('''vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
X_train = vec.fit_transform(X_train_text)                
X_test = vec.transform(X_test_text)   

X_train.nnz / (X_train.shape[0] * X_train.shape[1])   
X_test.nnz / (X_test.shape[0] * X_test.shape[1])          
type(X_train)                ''')
        
        
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
        X_train = vec.fit_transform(X_train_text)
        X_test = vec.transform(X_test_text)
        
        st.code(X_train.nnz / (X_train.shape[0] * X_train.shape[1]), language='html')
        st.code(X_test.nnz / (X_test.shape[0] * X_test.shape[1]), language='html')
        st.write(type(X_train))
      
      
        st.write('---')
        
        st.write('##### Gaussian NB')

        st.code('''vec1 = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=5)
X_train1 = vec1.fit_transform(X_train_text)
X_test1= vec1.transform(X_test_text)

X_train1.shape''')

        vec1 = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=5)
        X_train1 = vec1.fit_transform(X_train_text)
        X_test1= vec1.transform(X_test_text)
        
        st.code(f'{X_train1.shape}', language='html')
        
        
        st.code('''def dataset_transversal(X, Y, partial_function):
    chunk_size = 5000                
    classes = np.unique(Y)  
    lower = 0
    
    for upper in iter(range(chunk_size, X.shape[0], chunk_size)):         
        partial_function(X[lower:upper], Y[lower:upper], classes)
        lower = upper
        
    partial_function(X[upper:], Y[upper:], classes)
    
    
gnb = GaussianNB_sk()  
dataset_transversal(X_train1, y_train, lambda x,y,c:gnb.partial_fit(x.toarray(),y,c))
dataset_transversal(X_test1, y_test, lambda x,y,c:print(gnb.score(x.toarray(),y)))  
                ''')
    
    

    
        
        
        def dataset_transversal(X, Y, partial_function):
            chunk_size = 5000
            classes = np.unique(Y)
            lower = 0
            for upper in iter(range(chunk_size, X.shape[0], chunk_size)):
                partial_function(X[lower:upper], Y[lower:upper], classes)
                lower = upper
                
            partial_function(X[upper:], Y[upper:], classes)
            
            
        gnb = GaussianNB_sk()
        #dataset_transversal(X_train1, y_train, lambda x,y,c:gnb.partial_fit(x.toarray(),y,c))
        #dataset_transversal(X_test1, y_test, lambda x,y,c:print(gnb.score(x.toarray(),y)))
      
      
        st.write('---')
        
        st.write('##### Bernoulli + CountVectorizer')
        
        st.code('''vec2 = CountVectorizer(stop_words='english', binary=True, ngram_range=(1,3)) 
X_train2 = vec1.fit_transform(X_train_text)
X_test2= vec1.transform(X_test_text)                    
                
bnb = BernoulliNB_sk()
bnb.fit(X_train2, y_train)
Score para BN Bernouli: {bnb.score(X_test2, y_test)}''')
        
        
        vec2 = CountVectorizer(stop_words='english', binary=True, ngram_range=(1,3)) 
        
        X_train2 = vec2.fit_transform(X_train_text)
        X_test2= vec2.transform(X_test_text)            
      
        bnb = BernoulliNB_sk()
        bnb.fit(X_train2, y_train)
        st.write(f'Score para BN Bernouli: {bnb.score(X_test2, y_test)}')
      
      
        st.write('---')
        
        st.write('##### TF-IDF + Multinommial')
        
        st.code('''vec3 = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
X_train3 = vec3.fit_transform(X_train_text)
X_test3 = vec3.transform(X_test_text)              
                
mnb = MultinomialNB_sk()
mnb.fit(X_train3, y_train)  
Score para BN Multinomial: {mnb.score(X_test3, y_test)}''')        
        
        
        vec3 = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
        X_train3 = vec3.fit_transform(X_train_text)
        X_test3 = vec3.transform(X_test_text)
        
        mnb = MultinomialNB_sk()
        mnb.fit(X_train3, y_train)        
        st.write(f'Score para BN Multinomial: {mnb.score(X_test3, y_test)}')      
      
        st.write('---')
        
        st.write('##### CountVectorizer + Complement')
        
        st.code('''vec4 = CountVectorizer(stop_words='english', ngram_range=(1,3))
X_train4 = vec4.fit_transform(X_train_text)               
X_test4 = vec4.transform(X_test_text)  

cnb = ComplementNB_sk()  
cnb.fit(X_train4, y_train)           
Score para BN Complement: {cnb.score(X_test4, y_test)}''')
        
        vec4 = CountVectorizer(stop_words='english', ngram_range=(1,3))
        X_train4 = vec4.fit_transform(X_train_text)
        X_test4 = vec4.transform(X_test_text)
        
        cnb = ComplementNB_sk()
        cnb.fit(X_train4, y_train)        
        st.write(f'Score para BN Complement: {cnb.score(X_test4, y_test)}')    
        
        
        st.write('---')
        
        st.write('##### Preprocesamiento')        
        
        st.code('''tfidVec = TfidfVectorizer(stop_words='english', min_df=10)
X_train5 = tfidVec.fit_transform(X_train_text)     

mnb2 = MultinomialNB_sk.fit(X_train5, y_train)''')
        
        tfidVec = TfidfVectorizer(stop_words='english', min_df=10)
        X_train5 = tfidVec.fit_transform(X_train_text)
        
        mnb2 = MultinomialNB_sk().fit(X_train5, y_train)
        
        
        st.code('''km = KMeans(n_clusters=1000, random_state=1)
feature_to_cluster = km.fit_predict(mnb2.feature_log_prob_.T)''')
        
        
        km = KMeans(n_clusters=1000, random_state=1)
        feature_to_cluster = km.fit_predict(mnb2.feature_log_prob_.T)
        
        st.code('type(feature_to_cluster)  # nparray')
        st.write(type(feature_to_cluster))              # nparray
        
        st.code('type(mnb2.feature_log_prob_.T) # nparray')
        st.write(type(mnb2.feature_log_prob_.T))        # nparray
        
        st.code('(mnb2.feature_log_prob_.T).shape # (14967,4) ')
        st.code((mnb2.feature_log_prob_.T).shape)      # (14967,4) 
        
        st.code('(feature_to_cluster).shape # (14967,)')
        st.code((feature_to_cluster).shape)            # (14967,)
        
        st.code('feature_to_cluster # indica el cluster al que pertenece cada elemento')
        st.code(feature_to_cluster)                    # indica el cluster al que pertenece cada elemento
        
        st.code('feats2cluster = OneHotEncoder().fit_transform(feature_to_cluster.reshape(-1,1))')
        
        feats2cluster = OneHotEncoder().fit_transform(feature_to_cluster.reshape(-1,1))
        
        st.code('type(feats2cluster))')
        st.write(type(feats2cluster))
        
        st.code('feats2cluster.shape # (14967, 1000)')
        st.code(feats2cluster.shape)       # (14967, 1000)
      
        st.code('feats2cluster = feats2cluster.toarray()')
        feats2cluster = feats2cluster.toarray()
        
        st.code('feats2cluster.shape # (14967, 1000)')
        st.code(feats2cluster.shape)    # (14967, 1000)
        
        st.code('feats2cluster')
        st.code(feats2cluster)
        
        st.code('lista_clus = pd.Series(feats2cluster.sum(axis=0))')
        lista_clus = pd.Series(feats2cluster.sum(axis=0))

        st.write('---')
      
        st.code('''fig,ax = plt.subplots()   
ax.hist(x=feature_to_cluster, bins='auto', edgecolor='#000000', color='#8b92cc', alpha=.8)
ax.set_title('Cantidad de palabras en cada cluster')
ax.set_xlabel('Nro de cluster')
ax.set_ylabel('Cantidad de palabras')
                
st.pyplot(fig)''')
      
        with st.container(width=800):
            fig,ax = plt.subplots()   
            ax.hist(x=feature_to_cluster, bins='auto', edgecolor='#000000', color='#8b92cc', alpha=.8)
            ax.set_title('Cantidad de palabras en cada cluster')
            ax.set_xlabel('Nro de cluster')
            ax.set_ylabel('Cantidad de palabras')

            st.pyplot(fig)
        
        st.write('---')    

        st.code('''fig,ax = plt.subplots() 
ax.scatter(x=lista_clus, y=[i+1 for i in range(0,1000)], color='#ff8c00')
ax.set_title('Cantidad de palabras en cada cluster')
ax.set_xlabel('Cantidad de palabras')
ax.set_ylabel('Nro de cluster')                
                
st.pyplot(fig)''')


        with st.container(width=800):
            fig,ax = plt.subplots()   
            ax.scatter(x=lista_clus, y=[i+1 for i in range(0,1000)], color='#ff8c00')
            ax.set_title('Cantidad de palabras en cada cluster')
            ax.set_xlabel('Cantidad de palabras')
            ax.set_ylabel('Nro de cluster')

            st.pyplot(fig)
        
        st.write('---')           

        st.code('len(tfidVec.vocabulary_) # 14967')
        st.code(len(tfidVec.vocabulary_))   # 14967
        
        st.code('tfidVec.vocabulary_')
        st.code(tfidVec.vocabulary_)
        
        st.code('tfidvec_pd = pd.DataFrame(tfidVec.vocabulary_.keys(), index=(tfidVec.vocabulary_).values(), columns=[\'words\'])')
        tfidvec_pd = pd.DataFrame(tfidVec.vocabulary_.keys(), index=(tfidVec.vocabulary_).values(), columns=['words'])
        
        st.code('tfidvec_pd.shape')
        st.code(tfidvec_pd.shape)
        
        st.code('tfidvec_pd.head()')
        st.code(tfidvec_pd.head())
        

        st.code('''vocab = tfidVec.vocabulary_
countvec = CountVectorizer(stop_words='english')                
countvec.vocabulary_ = vocab                

X_train6 = countvec.transform(X_train_text)
X_test6 = countvec.transform(X_test_text)''')
      
        vocab = tfidVec.vocabulary_
        countvec = CountVectorizer(stop_words='english')
        countvec.vocabulary_ = vocab
        
        X_train6 = countvec.transform(X_train_text)
        X_test6 = countvec.transform(X_test_text)
      

        st.code('type(X_train6.shape)')
        st.code(type(X_train6.shape))
        
        st.code('feats2cluster.shape')
        st.code(feats2cluster.shape)
                
        st.code('''X_train_cluster = (X_train6 @ feats2cluster)
X_test_cluster = (X_test6 @ feats2cluster)''')        
                
        X_train_cluster = (X_train6 @ feats2cluster)
        X_test_cluster = (X_test6 @ feats2cluster)
      
        st.code('type(X_train_cluster)')
        st.code(type(X_train_cluster))
        
        st.code('X_train_cluster.shape')
        st.code(X_train_cluster.shape)
        
        st.code('X_train_cluster')
        st.code(X_train_cluster)
      

        st.code('''X_train_cluster[X_train_cluster > 2] = 2
X_test_cluster[X_test_cluster > 2] = 2 ''')
        X_train_cluster[X_train_cluster > 2] = 2
        X_test_cluster[X_test_cluster > 2] = 2 
        
        
        
        st.code('''cnb = CategorialNB_sk().fit(X_train_cluster, y_train)
st.code(f'Score de Categorical NB {cnb.score(X_test_cluster, y_test)}')''')
        cnb = CategorialNB_sk().fit(X_train_cluster, y_train)
        st.code(f'Score de Categorical NB {cnb.score(X_test_cluster, y_test)}')
        
        
        
        
      
      
    if opcion_seleccionada == 'Vinos':
        st.write('##### Data Frame con los datos de vino de 3 viñedos diferentes')  
        st.write('El algoritmo trata de determinar de que viñedo es un vino.')
      
        st.code('''from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report''')
        
        
        from sklearn.preprocessing import StandardScaler
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.metrics import classification_report
      
      
        st.code('''# load dataset
dataset = datasets.load_wine()   
dataset''')
        # load dataset
        dataset = datasets.load_wine()
        st.write(dataset)
         
        # names of the features
        st.write(f'**Inputs**: {dataset.feature_names}')
        st.write(f'**Outputs**: {dataset.target_names}')
        st.write(f'**Target**: {dataset.target}')

        st.write('---')
        st.write('###### Separación de los datos del modelo')

        st.code('''X = dataset.data
y = dataset.target   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1115, stratify=y)''')
        
        
        X = dataset.data
        y = dataset.target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1115, stratify=y)
    

        st.code('''df_vinos = pd.DataFrame(dataset.data, columns=dataset.feature_names)
st.dataframe(df_vinos(10))''')
        
        df_vinos = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        st.dataframe(df_vinos.head(10))

        st.write('---')
  
        st.write('##### Escalamiento de Datos')
        st.write(''' * **StandardScaler**: es una herrmaienta de preprocesamiento de datos que se utiliza para estandarizar funciones eliminando la media y escalando a la varianza media.  
Muchos algoritmos de ML funcionan mejor o convergen más rápido cuando las funciones están en una escala similar y centradas alrededor de cero.              
StandardScaler aborda esto transformando los datos de modo que cada característica tenga una media de 0 y una desviación estándar de 1.         
* **fit(data)** : se utiliza para calcular la media y la desviación estándar de una característica determinada que se utilizará posteriormente para escalar.          
* **transform(data)**: se utiliza para realizar el escalamiento utilizando la media y la desviación estándar calculadas utilizando el método .fit()''')
      
        st.code('''scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)                
X_test = scaler.transform(X_test)''')
      
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
      
        st.write('---')
        st.write('##### Entrenamiento del modelo') 
               
        st.code('''# Crear Clasificador Gaussiano
classifier = GaussianNB()                
classifier.fit(X_train, y_train)

# Prediccion                
y_pred = classifier.predict(X_test)''')       
               
        # Crear Clasificador Gaussiano
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        # Prediccion
        y_pred = classifier.predict(X_test)
        
        # Accuracy
        st.code(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')

        st.write('---')
        # Matriz de confusion
        st.write('##### Matriz de Confusion')
        with st.container(width=300):
            st.code('confusion_matrix(y_test, y_pred)')
            st.write(confusion_matrix(y_test, y_pred))
      
      
        st.write('---')
        # Reporte de Clasificacion
        st.write('##### Reporte de Clasificacion')      

        st.code('classification_report(y_test, y_pred, output_dict=True)')
        st.dataframe(classification_report(y_test, y_pred, output_dict=True))
      
      
      
    if opcion_seleccionada == 'Mensajes SMS':
        st.write('##### Data Frame con los datos de Mensajes SMS') 
         
        st.write('El algoritmo trata de determinar si un mensaje es Spam o no.')
        
        
        from zipfile import ZipFile
        
      #  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
       # filename = url.split('/')[-1]
       # fil = ZipFile(filename, 'r').namelist()[0]
      #  df_sms = pd.read_table(fil, header=0, names=['type','message'])
        
       # st.dataframe(df_sms.head(20))
        
        st.code('''filename = 'DataFrames\SMSSpamCollection'
df_sms = pd.read_table(filename, header=0, names=['type','message'])          
df_sms.head(10)''')
        
        
        filename = 'DataFrames\SMSSpamCollection'
        df_sms = pd.read_table(filename, header=0, names=['type','message'])
        
        st.dataframe(df_sms.head(10))
        
        st.code('''# info
df_college.info()''') 
        
        df_sms.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')  

        st.write('---')        
        
        st.code('''import nltk
nltk.download('punkt_tab')                
nltk.download('stopwords')   

from nltk.stem.porter import PorterStemmer    
from nltk.corpus import stopwords  
stop = list(stopwords.words('english'))''')
        
        import nltk
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        
       # from nltk.stem.porter import *
        from nltk.stem.porter import PorterStemmer
        from nltk.corpus import stopwords
        stop = list(stopwords.words('english'))
        
        # Agregar mas stopwords
        st.code('''# Agregar mas palabras a la lista stop
mas_palabras = ['.','*',',','?']
stop += mas_palabras ''')
        
        
        mas_palabras = ['.','*',',','?']
        stop += mas_palabras 
    
        
        
        st.code('''# Tokenize
df_sms['tokens'] = df_sms.apply(lambda x:nltk.word_tokenize(x['message']), axis=1)                
df_sms.head(10)''')        
        
        # Tokenize
        df_sms['tokens'] = df_sms.apply(lambda x:nltk.word_tokenize(x['message']), axis=1)
        st.dataframe(df_sms.head(10))
        
        st.write('---')
        
        st.code('''# Remover stop words
df_sms['tokens'] = df_sms['tokens'].apply(lambda x: [item for item in x if item not in stop])               
df_sms.head(10)''')
        
        # Reomver stop words
        df_sms['tokens'] = df_sms['tokens'].apply(lambda x: [item for item in x if item not in stop])
        st.dataframe(df_sms.head(10))
        
        st.write('---')
        
        st.code('''# Recortar terminaciones
stemmer = PorterStemmer()   
df_sms['tokens'] = df_sms['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])            
df_sms.head(10)                ''')
        
        # Recortar terminaciones
        stemmer = PorterStemmer()
        df_sms['tokens'] = df_sms['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])
        st.dataframe(df_sms.head(10))
        
        st.write('---')
        
        
        st.code('''from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Unificar de nuevo los strings
df_sms['tokens'] = df_sms['tokens'].apply(lambda x: ' '.join(x))

# Realizar split
X = df_sms['tokens']
y = df_sms['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Crear vectorizador
vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True)
''')
        
        
        
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Unificar de nuevo los strings
        df_sms['tokens'] = df_sms['tokens'].apply(lambda x: ' '.join(x))
        
        # Realizar split
        X = df_sms['tokens']
        y = df_sms['type']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=df_sms['type'])
        
        # Crear vectorizador
        vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True)
     
        # Entrenamiento y tranforacion
        
        st.code('''# Entrenamiento y transformacion
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)''')
        
        X_train_transformed = vectorizer.fit_transform(X_train)
        X_test_transformed = vectorizer.transform(X_test)
        
        st.write('---')
        st.write('**Entrenamiento del modelo**')
        
        st.code('''from sklearn.naive_bayes import MultinomialNB
                
naive_bayes = MultinomialNB()                
naive_bayes_fit = naive_bayes.fit(X_train_transformed, y_train)''')
        
        st.write('---')
        # Construccion del modelo
        from sklearn.naive_bayes import MultinomialNB
        
        # Entrenamiento del modelo
        naive_bayes = MultinomialNB()
        naive_bayes_fit = naive_bayes.fit(X_train_transformed, y_train)
        
 
        
        st.write('**Realizar predicciones**')
        
        st.code('''train_predict = naive_bayes_fit.predict(X_train_transformed)
test_predict = naive_bayes_fit.predict(X_test_transformed)''')
        
        train_predict = naive_bayes_fit.predict(X_train_transformed)
        test_predict = naive_bayes_fit.predict(X_test_transformed)
        
        st.write('---')
        
        st.write('**Matriz de Confusión**')
        
        st.code('''from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
                
def get_scores(y_real, predict):                
    ba_train = balanced_accuracy_score(y_real, predict)                
    cm_train = confusion_matrix(y_real, predict, normalize='all') 
    ConfusionMatrixDisplay(cm_train, display_labels=naive_bayes_fit.classes_).plot()               
    return ba_train, cm_train
    
def print_scores(scores):    
    return f'Balanced Accuracy: {scores[0]} Confusion Matriz: {scores[1]}''')
        
        
        
        
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
        

        def get_scores(y_real, predict):
            ba_train = balanced_accuracy_score(y_real, predict)
            cm_train = confusion_matrix(y_real, predict, normalize='all')
            ConfusionMatrixDisplay(cm_train, display_labels=naive_bayes_fit.classes_).plot()
            
            return ba_train, cm_train
        
        def print_scores(scores):
            return f'Balanced Accuracy: {scores[0]}\nConfusion Matriz:\n {scores[1]}'
        
        
        st.code('''train_scores = get_scores(y_train, train_predict)
test_scores = get_scores(y_test, test_predict)''')
        
        train_scores = get_scores(y_train, train_predict)
        test_scores = get_scores(y_test, test_predict)
        
        st.code('''st.code(print_scores(train_scores), language='html')
st.code(print_scores(test_scores), language='html')''')
        
        
        st.write('###### Train Score')
        st.code(print_scores(train_scores), language='html')
        st.write('###### Test Scores')
        st.code(print_scores(test_scores), language='html')
        
        
        
def ml_PCA():     
    buffer = io.StringIO()   

    st.write('#### Definición')
    st.write('''Una máquina de vectores de soporte (SVM) es una algoritmo de aprendizaje supervisado para clasificación y regresión, que funciona encontrando el **hiperplano óptimo**      
que separa las clases de datos con el máximo margen posible, es decir, la mayor distancia entre el hiperplano y los puntos más cercanos de cada clase (los vectores de soporte).        
Es muy eficaz en espacios de alta dimensionalidad y utiliza 'trucos de kernel' para manejar datos no linealmente separables, transformándolos a un espacio superior donde sí lo son.''')

    st.write('---')          
        
        
    opciones_mlpca = ['Cancer']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlpca)
        st.success(f'##### **{opcion_seleccionada}** ')

      
    if opcion_seleccionada == 'Cancer':
        st.write('##### Data Frame con los datos de Cancer')  
        st.code('''# Carga de csv con los datos de Cancer para deterinar si es maligno o benigno
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()''')  
        
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        
        st.code('cancer.keys()')
        st.code(cancer.keys(), language='html')
        
        st.code('df_cancer = pd.DataFrame(cancer[\'data\'], columns=cancer[\'feature_names\'])')
        
        df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])    
        st.write(df_cancer.head(10))    

        st.write('**Cancer Target**')
        st.code('''cancer['target']
cancer['target_names']''')

        st.code(cancer['target'], language='html')
        st.code(cancer['target_names'], language='html')       
        
        st.write('''Mediante PCA se buscan los dos componentes principales y visualizar los datos en este nuevo espacio dimensional.''')
        st.write('---')
        
        st.write('##### Escalamiento de datos')
        
        st.write('''* **StandardScaler**: es una herrmaienta de preprocesamiento de datos que se utiliza para estandarizar funciones eliminando la media y escalando a la varianza media.     
Muchos algoritmos de ML funcionan mejor o convergen más rápido cuando las funciones están en una escala similar y centradas alrededor de cero.       
StandardScaler aborda esto transformando los datos de modo que cada característica tenga una media de 0 y una desviación estándar de 1.
* **fit(data)**: se utiliza para calcular la media y la desviación estándar de una característica determinada que se utilizará posteriormente para escalar.
* **transform(data)**: se utiliza para realizar el escalamiento utilizando la media y la desviación estándar calculadas utilizando el método .fit()
''')
        
        st.code('''from sklearn.preprocessing import StandardScaler
                
scaler = StandardScaler()    
scaler.fit(df_cancer)
scaled_data = scaler.transform(df_cancer)
''')
        
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        scaler.fit(df_cancer)
        scaled_data = scaler.transform(df_cancer)
        
        st.write('---')
        
        st.code('''from sklearn.decomposition import PCA 
                
pca = PCA(n_components=2)                
pca.fit(scaled_data
x_pca = pca.transform(scaled_data)
x_pca.shape ''')
        
        # PCA
        from sklearn.decomposition import PCA 
        pca = PCA(n_components=2)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)
        
        st.code(x_pca.shape, language='html')
        
        st.code('''fig,ax = plt.subplots()  
    ax.scatter(x=x_pca[:,0], y=x_pca[:,1], c=cancer['target'], cmap='plasma')
    ax.set_title('Diagrama de Dispersión')
    ax.set_xlabel('Primer componente principal')
    ax.set_ylabel('Segundo componente principal') 
    
    st.pyplot(fig)''')
       

        with st.container(width=800):
            fig,ax = plt.subplots()   
            ax.scatter(x=x_pca[:,0], y=x_pca[:,1], c=cancer['target'], cmap='plasma')
            ax.set_title('Diagrama de Dispersión')
            ax.set_xlabel('Primer componente principal')
            ax.set_ylabel('Segundo componente principal')

            st.pyplot(fig) 

        
        
                
def ml_SVM():
    buffer = io.StringIO()   

    st.write('#### Definición')
    st.write('''Una máquina de vectores de soporte (SVM) es una algoritmo de aprendizaje supervisado para clasificación y regresión, que funciona encontrando el **hiperplano óptimo**      
que separa las clases de datos con el máximo margen posible, es decir, la mayor distancia entre el hiperplano y los puntos más cercanos de cada clase (los vectores de soporte).        
Es muy eficaz en espacios de alta dimensionalidad y utiliza 'trucos de kernel' para manejar datos no linealmente separables, transformándolos a un espacio superior donde sí lo son.''')

    st.write('---')    
    
    opciones_mlSVM = ['Breast_cancer', 'iris']

    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar:', opciones_mlSVM)
        st.success(f'#### **{opcion_seleccionada}**')
            
        @st.cache_data
        def load_svm_cancer():
            
            st.code('''from sklearn.datasets import load_breast_cancer  
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])                     
df_target = pd.DataFrame(cancer['target'])''')
            from sklearn.datasets import load_breast_cancer        
            cancer = load_breast_cancer()        

            df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
            df_target = pd.DataFrame(cancer['target'])
            
            return df_cancer, df_target
        
        @st.cache_data
        def load_svm_iris():
            df_iris = pd.read_csv('DataFrames/iris.csv', index_col='Id')
            
            return df_iris
                
                
    if opcion_seleccionada == 'iris':
        st.write(st.write('##### Data Frame con los datos de flores')  )            
        st.write('Se intenta predecir la especie')     
                
        df_iris = load_svm_iris()
        st.dataframe(df_iris.head(10))
         
        st.code('''# info
df_iris.info()''') 
        
        df_iris.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          

        st.write('---')
        st.write('##### Pairplot de las Species')

        st.code('''graf = sns.pairplot(
    data = df_iris,
    hue= 'Species',
    corner= True,
    palette='rocket_r)''')
    
        graf = sns.pairplot(
        data = df_iris,
        hue= 'Species',
        corner= True,
        palette='rocket_r')

        st.pyplot(graf)


        st.write('---')
        st.write('##### Seperación de los datos del modelo')           
                
        st.code('''X = df_iris.drop('Species', axis=1)
y = df_iris['Species']''')        
        X = df_iris.drop('Species', axis=1)
        y = df_iris['Species']
        
    
        st.write('---')
        st.write('##### Entrenamiento del modelo')  
        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.)
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')

        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)''')
    
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)      
    
                
        st.code('''from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)                
''')
        
        from sklearn.svm import SVC
        model = SVC()
        model.fit(X_train, y_train)
        
        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado. Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.')

        st.code('predictions = model.predict(X_test)')
        
        predictions = model.predict(X_test)
        
        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report 
st.dataframe(classification_report(y_test, predictions, output_dict=True))''')                
                
        from sklearn.metrics import classification_report     
        st.dataframe(classification_report(y_test, predictions, output_dict=True))
        
        st.write('---')
        st.write('##### Matrix de Confusión')
        
        
        st.code('''from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)''')
        
        with st.container(width=300):
            from sklearn.metrics import confusion_matrix
            st.dataframe(confusion_matrix(y_test, predictions)) 
 
 
 
 
 
                
                
    if opcion_seleccionada == 'Breast_cancer':
        st.write('##### Data Frame con los datos de tumores')  
        st.write('Se intenta predecir si el tumor es beningo o maligno')
        
        df_cancer, df_target = load_svm_cancer()
        st.dataframe(df_cancer.head(10))


        st.code('''# info
df_cancer.info()''') 
        
        df_cancer.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          

        st.write('---')
        st.write('##### Seperación de los datos del modelo')  
        
        st.code('''X = df_cancer
y = df_target                
''')
        X = df_cancer
        y = df_target
    
        st.write('---')
        st.write('##### Entrenamiento del modelo')  
        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.)
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')

        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)''')
    
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)    

        st.code('''from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)                
''')
        
        from sklearn.svm import SVC
        model = SVC()
        model.fit(X_train, y_train)
        
        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado. Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.')

        st.code('predictions = model.predict(X_test)')
        
        predictions = model.predict(X_test)
        
        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report 
st.dataframe(classification_report(y_test, predictions, output_dict=True))''')
        
        from sklearn.metrics import classification_report     
        st.dataframe(classification_report(y_test, predictions, output_dict=True))
        
        st.write('---')
        st.write('##### Matrix de Confusión')
        
        
        st.code('''from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)''')
        
        with st.container(width=300):
            from sklearn.metrics import confusion_matrix
            st.dataframe(confusion_matrix(y_test, predictions))
        
        
        #from sklearn import GridSearchCV
        #param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
        
        #grid = GridSearchCV(SVC(), param_grid, verbose=3)
        #grid.fit(X_train, y_train)
        
        
        
        
        
def ml_ensambletrees():
    buffer = io.StringIO()   

    st.write('#### Definición')
    st.write('''Un modelo de conjunto de árboles de decisión es una técnica de aprendizaje automático que combina múltiples ábroles de decisión para generar mejores predicciones o clasificaciones.    
Cada árbol de decisión del conjunto funciona como un simple sistema de reglas de ¨si-entonces¨.
Toma los datos de entrada y los divide en grupos más pequeños según diferentes características.     
Cada división crea ramas, y al final de estas ramas se encuentran las predicciones o clasificaciones.''')
    st.write('---')    
    
    opciones_mltrees = ['Car', 'Iris']           
        
    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mltrees)
        st.success(f'##### **{opcion_seleccionada}** ')

        @st.cache_data
        def load_data_car():
            df_Car = pd.read_csv('DataFrames/car_data', names=['buying','maint','doors','persons','lug_boot','safety','acceptability'])
            
            return df_Car        

        @st.cache_data
        def load_data_iris():
            df_iris = pd.read_csv('DataFrames/Iris.csv',index_col='Id')
            
            return df_iris  


    if opcion_seleccionada == 'Iris':
        st.write('##### Data Frame con datos de Iris')     
        
        st.code('''df_iris = pd.read_csv('DataFrames/Iris.csv',index_col='Id')
df_Iris.head(15)''')
        
        df_Iris = load_data_iris()
        st.dataframe(df_Iris.head(15))        

        st.code('''# info
df_Iris.info()''') 
            
        df_Iris.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')  

        st.write('---')        
        
        st.code('''X = df_Iris[['PetalLengthCm','PetalWidthCm']]  
y = df_Iris['Species']''')
        
        # Comversion species a numerico
        st.write('---')
        st.write('Conversion de campo Species a numerica')

        st.code('''data = {
'Iris-setosa':0,                
'Iris-versicolor':1,   
'Iris-virginica':2             
}

df_Iris['Species'] = df_Iris['Species'].map(data)

X = df_Iris[['PetalLengthCm','PetalWidthCm']]
y = df_Iris['Species']''')

        data = {
            'Iris-setosa':0,
            'Iris-versicolor':1,
            'Iris-virginica':2
        }
        
        df_Iris['Species'] = df_Iris['Species'].map(data) 
        
        X = df_Iris[['PetalLengthCm','PetalWidthCm']]
        y = df_Iris['Species']
        
        st.write('---')
        
        st.code('''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier                
from sklearn.naive_bayes import GaussianNB                
from sklearn.ensemble import RandomForestClassifier    
from sklearn.model_selection import GridSearchCV            
from mlxtend.classifier import StackingClassifier''')
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        from mlxtend.classifier import StackingClassifier
        
        # Inicializacion de modelos
        
        st.code('''# Inicializacion de modelos  
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)               
clf3 = GaussianNB()                
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)   

params = {'kneighborsclassifier__n_neighbors': [1,5], 'randomforestclassifier__n_estimators': [10,50], 'meta_classifier__C': [0.1,10.0]}   

grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
grid.fit(X,y)
cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    st.write(f'%0.3f +/- %0.2f %r'
        % (grid.cv_results_[cv_keys[0]][r],
        grid.cv_results_[cv_keys[1]][r]/2.0,
        grid.cv_results_[cv_keys[2]][r]))
''') 
        
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = GaussianNB()
        lr = LogisticRegression()
        sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

        params = {'kneighborsclassifier__n_neighbors': [1,5], 'randomforestclassifier__n_estimators': [10,50], 'meta_classifier__C': [0.1,10.0]}
        
        grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)    
    
        grid.fit(X,y)
        
        cv_keys = ('mean_test_score', 'std_test_score', 'params')
        
        for r, _ in enumerate(grid.cv_results_['mean_test_score']):
            st.write(f'%0.3f +/- %0.2f %r'
                    % (grid.cv_results_[cv_keys[0]][r],
                       grid.cv_results_[cv_keys[1]][r]/2.0,
                       grid.cv_results_[cv_keys[2]][r]))
        
        st.write('---')
        st.write('Best parameters: %s' % grid.best_params_)
        st.write('Accuracy: %.2f' % grid.best_score_)
        
        
        
        
    if opcion_seleccionada == 'Car':
        st.write('##### Data Frame con datos de Autos')     
        
        st.code('''df_Car = pd.read_csv('DataFrames/car_data', names=['buying','maint','doors','persons','lug_boot','safety','acceptability'])
df_Car.head(15)''')
        
        df_Car = load_data_car()
        st.dataframe(df_Car.head(15))


        st.code('''# info
df_Car.info()''') 
            
        df_Car.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')  

        st.write('---')
        
        st.write('**Utilizacion de LabelEnconder para convertir la columna categorica Acceptability en numerica, y get_dummies para convertir las demas columnas a numericas.**') 
        
        st.code('''from sklearn.preprocessing import LabelEncoder
                
le = LabelEncoder()
le.fit(df_Car['acceptability']) ''')
        
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        le.fit(df_Car['acceptability'])        
        
        st.code(le.classes_, language='html')
        
        st.code('''y = le.transform(df_Car['acceptability'])
X = pd.get_dummies(df_Car.drop('acceptability', axis=1), dtype=int, drop_first=True) ''')  
        
        
        y = le.transform(df_Car['acceptability'])
        X = pd.get_dummies(df_Car.drop('acceptability', axis=1), dtype=int, drop_first=True)
        
        st.dataframe(X.head())

        st.write('---')
        st.write('**Separacion de los datos del modelo**')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=41, stratify=y)
        
        st.write('Para que los resultados sean consistentes hay que exponer los modelos exactamente al mismo esquema de validazion cruzada.')
        
        
        st.code('''from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=3, random_state=41, shuffle=True)                

def evaluar_rendimiento(modelo, nombre, X_train, y_train, cv):
    s = cross_val_score(modelo, X_train, y_train, cv=cv, n_jobs=1)
    st.write(f'Rendimiento de {nombre}: {s.mean().round(3):0.3} {s.std().round(3)}')''')
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        cv = StratifiedKFold(n_splits=3, random_state=41, shuffle=True)
        
        def evaluar_rendimiento(modelo, nombre, X_train, y_train, cv):
            s = cross_val_score(modelo, X_train, y_train, cv=cv, n_jobs=1)
            st.write(f'Rendimiento de {nombre}: {s.mean().round(3):0.3} +/- {s.std().round(3):0.3}')
        
        
        st.code('''from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=1), n_estimators=100)
gb = GradientBoostingClassifier()

evaluar_rendimiento(ab, 'AdaBoostClassifier', X_train, y_train, cv)
evaluar_rendimiento(gb, 'GradientBoostingClassifier', X_train, y_train, cv)''')
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
        
        ab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=1), n_estimators=100)
        gb = GradientBoostingClassifier()
        evaluar_rendimiento(ab, 'AdaBoostClassifier', X_train, y_train, cv)
        evaluar_rendimiento(gb, 'GradientBoostingClassifier', X_train, y_train, cv)
        
        st.write('---')
        st.write('El AdaBoost performa peor que el GradientBoost. Se modifican parametros para hacerlo funcionar mejor.')
        
        st.code('''from sklearn.model_selection import GridSearchCV

params_ab = {'n_estimators': [100,500], 'learning_rate': [0.01, 0.1, 1.0], 'estimator__max_depth': [1,2,3]}                 
grid_ab = GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier()), param_grid=params_ab, cv=cv, verbose=1, n_jobs=1)
grid_ab.fit(X_train, y_train)
mejor = grid_ab.best_estimator_
grid_ab.best_params_''')
        
        from sklearn.model_selection import GridSearchCV
        
        params_ab = {'n_estimators': [100,500], 'learning_rate': [0.01, 0.1, 1.0], 'estimator__max_depth': [1,2,3]}
        
        grid_ab = GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier()), param_grid=params_ab, cv=cv, verbose=1, n_jobs=1)
        grid_ab.fit(X_train, y_train)
        mejor = grid_ab.best_estimator_

        grid_ab.best_params_
        
        st.code('''evaluar_rendimiento(mejor, 'AdaBoostClassifier + G5', X_train, y_train, cv)''')
        
        evaluar_rendimiento(mejor, 'AdaBoostClassifier + GS', X_train, y_train, cv)
        
        st.write('---')
        st.write('Se modifican parametros para GradientBoost.')
        
        st.code('''params_gb = {'n_estimators': [100,500], 'learning_rate': [0.001, 0.01, 0.1, 1], 'max_depth': [1,2,3,4]}
grid_gb = GridSearchCV(gb, param_grid=params_gb, cv=cv, verbose=1, n_jobs=3)
grid_gb.fit(X_train, y_train)
mejor_gb = grid_gb.best_estimator_''')
        
        params_gb = {'n_estimators': [100,500], 'learning_rate': [0.001, 0.01, 0.1, 1.0], 'max_depth': [1,2,3,4]}
        grid_gb = GridSearchCV(gb, param_grid=params_gb, cv=cv, verbose=1, n_jobs=3)
        grid_gb.fit(X_train, y_train)
        mejor_gb = grid_gb.best_estimator_
        
        st.code('''evaluar_rendimiento(mejor_gb, 'GradientBoostingClassifier + GS', X_train, y_train, cv)''')
        
        evaluar_rendimiento(mejor_gb, 'GradientBoostingClassifier + GS', X_train, y_train, cv)
        
        st.write('---')
        st.write('**Valor de AUC y grafico de ROC**')
        
        st.code('''from sklearn.metrics import roc_auc_score
                
gb_auc = roc_auc_score(y_test, grid_gb.predict_proba(X_test), multi_class='ovr')                
st.write(f'El valor del AUC es: {gb_auc}')''')
        
        
        st.code('''from sklearn.metrics import roc_auc_score
                
gb_auc = roc_auc_score(y_test, grid_gb.predict_proba(X_test), multi_class='ovr')                
st.write(f'El valor del AUC es: {gb_auc}')                
import scikitplot     
from scikitplot.metrics import plot_roc      

fig,ax = plt.subplots() 
plot_roc(y_test, grid_gb.predict_proba(X_test))                

st.pyplot(fig)''')
        
        from sklearn.metrics import roc_auc_score
        
        gb_auc = roc_auc_score(y_test, grid_gb.predict_proba(X_test), multi_class='ovr')
        st.write(f'El valor del AUC es: {gb_auc}')
        
        
        import scikitplot
        from scikitplot.metrics import plot_roc 
        
        with st.container(width=600):
            fig,ax = plt.subplots()  
            plot_roc(y_test, grid_gb.predict_proba(X_test))
            
            st.pyplot(fig)
        
        st.code('''model = grid_gb.best_estimator_
model''')
    
        model = grid_gb.best_estimator_
        st.write(model)
        
        st.code('''importances = model.feature_importances_
importances * 100''')
        
        importances = model.feature_importances_
        st.code(importances * 100)
        
        st.code('''indices = np.argsort(importances)[::-1]
names = X.columns[indices]      

fig,ax = plt.subplots()   
ax.bar(range(X.shape[1]), importances[indices])
ax.set_title('Feature Importance')
ax.set_xticks(range((X.shape[1]), names, rotation=90))

st.pyplot(fig)''')
        
        
        # Crear variable que tenga los indices indicando los valores de mayor a menor
        indices = np.argsort(importances)[::-1]
        # con dicha variable realizamos fancy indexing de manera de ordenar los labels del eje x.
        names = X.columns[indices]
        
        fig,ax = plt.subplots()   
        ax.bar(range(X.shape[1]), importances[indices])
        ax.set_title('Feature Importance')
        ax.set_xticks(range((X.shape[1]), names, rotation=90))

        st.pyplot(fig)
        
        
        
        
        
        
        
        
        
def ml_trees():
    buffer = io.StringIO()   

    st.write('#### Definición')
    st.write('''Un árbol de decisión (Decision tree) es un modelo de Aprendizaje Automático que toma decisiones a través de un diagrama de árbol,   
mientras que un bosque aleatorio (Random forest) es un modelo de conjunto que utiliza múltiples árboles de decisión para lograr predicciones más precisas y robustas.   
El bosque aleatorio combina las predicciones de varios árboles entrenados con diferentes subconjuntos de datos y característicasm lo que hace menos propenso al sobreajuste (overfitting) que un solo árbol.''')
    st.write('---')    
    
    opciones_mltrees = ['Kyphosis', 'Prestamos', 'Pinguinos', 'Titanic']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mltrees)
        st.success(f'##### **{opcion_seleccionada}** ')

        @st.cache_data
        def load_data_trees():
            df_kyphosis = pd.read_csv('DataFrames/kyphosis.csv')
            
            return df_kyphosis
        
        @st.cache_data
        def load_date_prestamos():
            df_prestamos = pd.read_csv('DataFrames/loan_data.csv')
            
            return df_prestamos
    
        @st.cache_data
        def load_data_pinguinos():
            df_pinguinos = pd.read_csv('DataFrames/penguins.csv')
            
            return df_pinguinos    
    
        @st.cache_data
        def load_data_titanic():
            df_titanic = pd.read_csv('DataFrames/titanic_train.csv')

            return df_titanic
    
    
    
    
    
    if opcion_seleccionada == 'Titanic':
        st.write('##### Data Frame con datos del naufragio del Titanic')  
    
        st.code('''from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay''')
        
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.tree import plot_tree
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay    
    
    
        st.code('''df_titanic = pd.read_csv('DataFrames/titanic_train.csv')
df_titanic.head(10)''')

    
        df_titanic = load_data_titanic()
        st.dataframe(df_titanic.head(10))    


        st.code('''# info
df_titanic.info()''') 
        
        df_titanic.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')  


        st.write('---')
        st.write('**Agrega columna para hombres adultos**')
        
        st.code('''def categorizar(fila):
    if fila['Sex'] == 'male' and fila['Age'] > 17:                
        return 1                
    else:
        return 0           
                
df_titanic['adult_male'] = df_titanic.apply(categorizar, axis=1)''')
        
        def categorizar(fila):
            if fila['Sex'] == 'male' and fila['Age'] > 17:
                return 1
            else:
                return 0
        
        
        df_titanic['adult_male'] = df_titanic.apply(categorizar, axis=1)
        
    
        st.write('---')
        st.write('**Agregar promedio de edad en valores faltantes**')
        
        st.code('''filtro = df_titanic['Sex'] == 'female'
promedio_fem = df_titanic[filtro]['Age'].median()                
filtro = df_titanic['Sex'] == 'male'                
promedio_male = df_titanic[filtro]['Age'].median()                 ''')
       
        
        filtro = df_titanic['Sex'] == 'female'
        promedio_fem = df_titanic[filtro]['Age'].median()
        filtro = df_titanic['Sex'] == 'male'
        promedio_male = df_titanic[filtro]['Age'].median()    
    

        st.write(f'Promedio de edad female: {promedio_fem}')
        st.write(f'Promedio de edad male: {promedio_male}')

        st.code('''def promedio_edad(fila):
    if pd.isna(fila['Age']):                
        if fila['Sex'] == 'male':
            return promedio_male
        else:
            return promedio_fem    
    else:    
        return fila['Age']''')

        def promedio_edad(fila):
            if pd.isna(fila['Age']):
                if fila['Sex'] == 'male':
                    return promedio_male
                else:
                    return promedio_fem
            else:
                return fila['Age']
        

        st.code('''df_titanic['Age'] = df_titanic.apply(promedio_edad, axis=1)
df_titanic.head(10)''')
        
        df_titanic['Age'] = df_titanic.apply(promedio_edad, axis=1)

        st.write('---')
        st.write('**Setear PassengerId como indice**')

        st.code('df_titanic.set_index(\'PassengerId\', inplace=True)') 
        df_titanic.set_index('PassengerId', inplace=True)
    
    
        st.write('---')
        st.write('**Eliminar columna Cabin, Name y Ticket**')
        
        st.code('df_titanic.drop([\'Cabin\',\'Name\',\'Ticket\'], axis=1, inplace=True)')
        df_titanic.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)

        st.write('---')
        st.write('**Convertir Sex a valor numerico**')
        
        st.code('''data = {\'male\':0,\'female\':1}
df_titanic[\'Sex\'] = df_titanic[\'Sex\'].map(data)''')

        
        data = {'male':0,'female':1}
        df_titanic['Sex'] = df_titanic['Sex'].map(data)
        
        
        st.write('---')
        st.write('**Convertir Embarked a valor numerico**')
        
        st.code('''data = {'S':0, 'C':1, 'Q':2}
df_titanic['Embarked'] = df_titanic['Embarked'].map(data)''')
        
        data = {'S':0, 'C':1, 'Q':2}
        df_titanic['Embarked'] = df_titanic['Embarked'].map(data)
        
        
        st.dataframe(df_titanic.head(10))

        st.write('---')
       
        col1, col2= st.columns([2,2])
    
        with col1:
            st.write('**Catplot de sobrevivientes**')
            
            st.code('''graf = sns.catplot(
    data = df_titanic,
    x='Survived',
    kind='count',
    palette='inferno'                    
)
st.pyplot(graf)''')
            
            with st.container(width=600):
                graf = sns.catplot(
                    data = df_titanic,
                    x='Survived',
                    kind='count',
                    palette='inferno'
                )

                st.pyplot(graf)    
            
        with col2:
            st.write('**Countplot de sobrevivientes por clase**')
            
            st.code('''fig,ax = plt.subplots()
    sns.countplot(
    data = df_titanic,
    x='Pclass',
    hue='Survived', 
    palette='Set1'                  
)
st.pyplot(graf)''')            
            
            with st.container(width=600):
                fig,ax = plt.subplots()
                sns.countplot(
                    data = df_titanic,
                    x='Pclass',
                    hue='Survived',
                    palette='Set1'
                )

                st.pyplot(fig)      
    
    
        st.write('---')
       
        col1, col2= st.columns([2,2])
    
        with col1:
            st.write('**Countplat de sobrevivientes por sexo**')
            
            st.code('''fig,ax = plt.subplots()
    sns.countplot(
    data = df_titanic,
    x='Sex',
    hue='Survived', 
    palette='Set1'                  
)
st.pyplot(graf)''') 
            
            with st.container(width=600):
                fig,ax = plt.subplots()
                sns.countplot(
                    data = df_titanic,
                    x='Sex',
                    hue='Survived',
                    palette='Set1'
                )

                st.pyplot(fig) 


        st.write('---')
        st.write('**Separación de datos del Modelo**')

        st.code('''X = df_titanic.drop('Survived', axis=1)
y = df_titanic['Survided']          
          
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=16, stratify=y)''')

        X = df_titanic.drop('Survived', axis=1)
        y = df_titanic['Survived']
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=16, stratify=y)


        st.write('---')
        st.write('**Entrenamiento del modelo**')
        
        st.code('''arbol = DecisionTreeClassifier(criterion='entropy', random_state=16)
arbol.fit(X_train, y_train)''')
        
        arbol = DecisionTreeClassifier(criterion='entropy', random_state=16)
        arbol.fit(X_train, y_train)


        st.write('---')
        st.write('**Ver que variables pesaron mas en las decisiones del arbol**')
        
        fi = arbol.feature_importances_
        st.code(fi)
        
        st.code('''fi = arbol.feature_importances_
                
fig,ax = plt.subplots()                
sns.barplot(
    x=fi, 
    y=X_train.columns,
    palette='Spectral'    
)
st.pyplot(fig)''')
        
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            sns.barplot(
                x=fi, 
                y=X_train.columns,
                palette='Spectral'
            )
            st.pyplot(fig)

        
        st.write('---')
        st.write('**Grafico del Arbol**')
        
        st.code('''list_features = list(X.columns)
fig,ax = plt.subplots()  
plot_tree(arbol, filled=True, rounded=True, feature_names=list_features, max_depth=3)   
              
st.pyplot(fig)                  ''')
        
        list_features = list(X.columns)
        fig,ax = plt.subplots()
        plot_tree(arbol, filled=True, rounded=True, feature_names=list_features, max_depth=3)    
        st.pyplot(fig)  
                    
        
        st.write('---')
        st.write('**Predicciones**')
        
        st.code('''y_pred = arbol.predict(X_test)  
                   
Score train: {arbol.score(X_train, y_train)}
Score test: {arbol.score(X_test, y_test)}''')
        
        y_pred = arbol.predict(X_test)       
        st.write(f'Score train: {arbol.score(X_train, y_train)}')
        st.write(f'Score test: {arbol.score(X_test, y_test)}')

        st.write('Se ve que el arbol está sobreajustado.')

        st.write('---')
        st.write('**Matriz de Confusion**')

        st.code('confusion_matrix(y_test, y_pred))')
        
        with st.container(width=250):      
            st.write(confusion_matrix(y_test, y_pred))



        st.write('---')
        st.write('**Correccion del sobreajuste**')

        st.write(f'Maxima profundidad del arbol: {arbol.get_depth()}')
        st.write(f'cantidad de hojas: {arbol.get_n_leaves()}')
        
        st.write('---')
        st.write('**Modificar profundidad del arbol**')
        
        
        st.code('''depth = [i+1 for i in range(20)]
scor_train = []      
scor_test = []
      
for dep in depth:      
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=dep, random_state=16) 
    tree.fit(X_train, y_train) 
    pred = tree.predict(X_test) 
    
    scor_train.append(tree.score(X_train, y_train))   
    scor_test.append(tree.score(X_test, y_test))
    
    fig,ax = plt.subplots()
    ax.scatter(x=depth, y=scor_train, marker='o', color='blue')   # grafica los datos 
    ax.scatter(x=depth, y=scor_test, marker='x', color='red')
    
    ax.set_title('Score arbol vs Profundidad')
    ax.set_xlabel('Profundidad arbol')
    ax.set_ylabel('Score')
    
st.pyplot(fig)''')
        
        
        depth = [i+1 for i in range(20)]
        scor_train = []
        scor_test = []
        
        for dep in depth:
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=dep, random_state=16)
            tree.fit(X_train, y_train)
            pred = tree.predict(X_test)
            
            scor_train.append(tree.score(X_train, y_train))
            scor_test.append(tree.score(X_test, y_test))
        
        
        with st.container(width=800):
            fig,ax = plt.subplots()   
            ax.scatter(x=depth, y=scor_train, marker='o', color='blue')   # grafica los datos
            ax.scatter(x=depth, y=scor_test, marker='x', color='red')
          
            ax.set_title('Score arbol vs Profundidad')
            ax.set_xlabel('Profundidad arbol')
            ax.set_ylabel('Score')

            st.pyplot(fig)
        
        
        st.write('---')
        st.write('**Modificar cantidad de muestras maxima en cada hoja**')        
        
        
        st.code('''samplex = [i+1 for i in range(20)]
scor_train1 = []                
scor_test1 = []                
                
for sam in samplex:
    tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=sam, random_state=16)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test) 
    
    scor_train1.append(tree.score(X_train, y_train))
    scor_test1.append(tree.score(X_test, y_test))
    
    fig,ax = plt.subplots()   
    ax.scatter(x=samplex, y=scor_train1, marker='o', color='blue')   # grafica los datos
    ax.scatter(x=samplex, y=scor_test1, marker='x', color='red')
    
    ax.set_title('Score arbol vs Muestras en hoja')  
    ax.set_xlabel('Muestras en hoja')         
    ax.set_ylabel('Score') 

    st.pyplot(fig)''')
        
        
        samplex = [i+1 for i in range(20)]
        scor_train1 = []
        scor_test1 = []
        
        for sam in samplex:
            tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=sam, random_state=16)
            tree.fit(X_train, y_train)
            pred = tree.predict(X_test) 
            
            scor_train1.append(tree.score(X_train, y_train))
            scor_test1.append(tree.score(X_test, y_test))       
        
        with st.container(width=800):
            fig,ax = plt.subplots()   
            ax.scatter(x=samplex, y=scor_train1, marker='o', color='blue')   # grafica los datos
            ax.scatter(x=samplex, y=scor_test1, marker='x', color='red')
          
            ax.set_title('Score arbol vs Muestras en hoja')
            ax.set_xlabel('Muestras en hoja')
            ax.set_ylabel('Score')

            st.pyplot(fig)
                
        
        st.write('---')
        st.write('**Impureza de las hojas**')
        
        st.code('''clas = DecisionTreeClassifier(criterion='entropy', random_state=16)
path = clas.cost_complexity_pruning_path(X_train, y_train)                 
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle='steps-post')
ax.set_xlabel('Effective alpha')
ax.set_ylabel('Total impurity of leaves')
ax.set_title('Total Impurity vs Effective alpha for training set')     

st.pyplot(fig)''')
        
        clas = DecisionTreeClassifier(criterion='entropy', random_state=16)
        path = clas.cost_complexity_pruning_path(X_train, y_train) 
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        
        with st.container(width=800):
            fig, ax = plt.subplots()
            ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle='steps-post')
            ax.set_xlabel('Effective alpha')
            ax.set_ylabel('Total impurity of leaves')
            ax.set_title('Total Impurity vs Effective alpha for training set')          
            
            st.pyplot(fig)
            
        
        st.code('''clfs = []
for ccp_alpha in ccp_alphas:                
    clf = DecisionTreeClassifier(random_state=16, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)        
    
st.write(f'Number of nodes in the last tree is {clfs[-1].tree_.node_count} with ccp_alpha {ccp_alphas[-1]}')''')
        
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=16, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)
            
        st.write(f'Number of nodes in the last tree is {clfs[-1].tree_.node_count} with ccp_alpha {ccp_alphas[-1]}')
        
        
        st.code('''clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]                

node_counts = [clf.tree_.node_count for clf in clfs]                
depth = [clf.tree_.max_depth for clf in clfs]  

fig, ax = plt.plots(2,1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle='steps-post')
ax[0].set_xlabel('alpha')
ax[0].set_ylabel('number of nodes')
ax[0].set_title('Number of nodes vs Alpha')
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle='steps-post')
ax[1].set_xlabel('alpha')
ax[1].set_ylabel('number of tree')
ax[1].set_title('Depth vs Alpha')

st.pyplot(fig)''')
        
        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]        

        with st.container(width=800):
            fig, ax = plt.subplots(2,1)
            ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle='steps-post')
            ax[0].set_xlabel('alpha')
            ax[0].set_ylabel('number of nodes')
            ax[0].set_title('Number of nodes vs Alpha')
            ax[1].plot(ccp_alphas, depth, marker='o', drawstyle='steps-post')
            ax[1].set_xlabel('alpha')
            ax[1].set_ylabel('number of tree')
            ax[1].set_title('Depth vs Alpha')

            st.pyplot(fig)

        
        st.code('''train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]                

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing scores')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
    
st.pyplot(fig)''')
        
        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        test_scores = [clf.score(X_test, y_test) for clf in clfs]
        
        with st.container(width=800):
            fig, ax = plt.subplots()
            ax.set_xlabel('alpha')
            ax.set_ylabel('accuracy')
            ax.set_title('Accuracy vs alpha for training and testing scores')
            ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
            ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
        
            st.pyplot(fig)
    

        st.code('''alfas = ccp_alphas
scor = np.array(test_scores)
tscor = np.array(train_scores)
dif = tscor-scor
alfa_score = pd.DataFrame({'alpha':alfas, 'score':scor, 'dif':dif})
indice = alfa_score['dif'].idxmin()
alfa_best = alfa_score['alpha'].iloc[indice]
        
st.write(f'Mejor alfa {alfa_best}')    
st.dataframe(alfa_score, width=400)''')
        
        
        alfas = ccp_alphas
        scor = np.array(test_scores)
        tscor = np.array(train_scores)
        dif = tscor-scor
        alfa_score = pd.DataFrame({'alpha':alfas, 'score':scor, 'dif':dif})
        indice = alfa_score['dif'].idxmin()
        alfa_best = alfa_score['alpha'].iloc[indice]
        
        st.write(f'Mejor alfa {alfa_best}')    
        st.dataframe(alfa_score, width=400)
    
    
        st.write('---')
        st.write('**Arbol con el mejor alfa, profundidad y muestras por hoja**')
        
        st.code('''arbol_alfa = DecisionTreeClassifier(criterion='entropy', random_state=16, ccp_alpha=alfa_best)
                
arbol_alfa.fit(X_train, y_train)
y_pred_alfa = arbol_alfa.predict(X_test)
        
st.write(f'Score train: {arbol_alfa.score(X_train, y_train)}')
st.write(f'Score test: {arbol_alfa.score(X_train, y_train)}')
              
confusion_matrix(y_test, y_pred_alfa)''') 
        
        arbol_alfa = DecisionTreeClassifier(criterion='entropy', random_state=16, ccp_alpha=alfa_best)
        arbol_alfa.fit(X_train, y_train)
        y_pred_alfa = arbol_alfa.predict(X_test)
        
        st.write(f'Score train: {arbol_alfa.score(X_train, y_train)}')
        st.write(f'Score test: {arbol_alfa.score(X_test, y_test)}')
    
        with st.container(width=250):            
            st.write(confusion_matrix(y_test, y_pred_alfa))
    
    
    
    
    if opcion_seleccionada == 'Pinguinos':
        st.write('##### Data Frame con los datos de Pinguinos')  
        st.write('Se intenta predecir la subespecie de los pinguinos')
        
        st.code('''from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay''')        
        
        
        
        st.code('''# Carga de csv con los datos de Pinguinos
df_pinguinos = pd.read_csv('DataFrames/df_pinguinos.csv')     
df_pinguinos.head(10)       # head''')  


        
        
        from sklearn.pipeline import make_pipeline
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.tree import plot_tree
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        import warnings

        warnings.filterwarnings('ignore')

        df_pinguinos = load_data_pinguinos()
        st.dataframe(df_pinguinos.head(10))    
    
        st.code('''# info
df_pinguinos.info()''') 
        
        df_pinguinos.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          
    
    
        st.write('---')
        st.write('**Eliminación de datos faltantes**')

        st.code('df_pinguinos.dropna(subset=[\'sex\'], inplace=True)')
        df_pinguinos.dropna(subset=['sex'], inplace=True)
 
        st.code(df_pinguinos.shape)

        st.write('---')
        st.write('**Valores unicos**')       

        st.code('''especies = df_pinguinos.value_counts('species')  # Especies
especies''')
        
        especies = df_pinguinos.value_counts('species')
        st.code(especies)

        st.code('''sexo = df_pinguinos.value_counts('sex')      # Sexo
sexo''')

        sexo = df_pinguinos.value_counts('sex')
        st.code(sexo)


        st.code('''islas = df_pinguinos.value_counts('island')      # Islas
islas''')

        islas = df_pinguinos.value_counts('island')
        st.code(islas)


        st.write('---')
        st.write('**Conversion de variables categorias en numericas**')
        
        st.code('''data = {
    'Adelie':1,
    'Gentoo':2,
    'Chinstrap':3
} ''')
        
        data = {
            'Adelie':1,
            'Gentoo':2,
            'Chinstrap':3
        } 
        
        st.code('df_pinguinos[\'species\'] = df_pinguinos[\'species\'].map(data)')
        
        df_pinguinos['species'] = df_pinguinos['species'].map(data)


        st.code('''data = {
    'male':1,
    'female':2,
} ''')
        
        data = {
            'male':1,
            'female':2,
        } 
        
        st.code('df_pinguinos[\'sex\'] = df_pinguinos[\'sex\'].map(data)')
        
        df_pinguinos['sex'] = df_pinguinos['sex'].map(data)

        st.code('''data = {
    'Biscoe':1,
    'Dream':2,
    'Torgersen':3
} ''')
        
        data = {
            'Biscoe':1,
            'Dream':2,
            'Torgersen':3
        } 
        
        st.code('df_pinguinos[\'island\'] = df_pinguinos[\'island\'].map(data)')
        
        df_pinguinos['island'] = df_pinguinos['island'].map(data)
    
        st.dataframe(df_pinguinos.head(15))

        st.write('---')
        st.write('**Separacion de los datos del modelo**')
        
        st.code('''X = df_pinguinos.drop('species', axis=True)
y = df_pinguinos['species']''')
         
        X = df_pinguinos.drop('species', axis=True)        
        y = df_pinguinos['species']
        
        st.code('X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=26, stratify=y)')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=26, stratify=y)
    
    
        st.write('---')
        st.write('###### Arbol como Regresor')
        st.write('Existe una relacion entre el peso y el largo de la aleta del pinguino.')
        
        st.code('''xr = X_train['body_mass_g'].values.reshape(-1,1)
yr = X_train['flipper_length_mm'].values.reshape(-1,1)''')
        
        xr = X_train['body_mass_g'].values.reshape(-1,1)
        yr = X_train['flipper_length_mm'].values.reshape(-1,1)
    
    
        st.write('---')
        st.write('**Entrenamiento**')

        st.code('''arbol = DecisionTreeRegressor(criterion='squared_error', random_state=16)   # Instancia del modelo
arbol.fit(xr, yr)   # Entrenamiento ''')
        
        arbol = DecisionTreeRegressor(criterion='squared_error', random_state=16)
        arbol.fit(xr, yr)         # Entrenamiento

        st.write('---')
        st.write('**Prediccion**')

        st.code('''test_xr = X_test['body_mass_g'].values.reshape(-1,1)
test_yr = X_test['flipper_length_mm'].values.reshape(-1,1)                
y_pred = arbol.predict(test_xr)''')

        test_xr = X_test['body_mass_g'].values.reshape(-1,1)
        test_yr = X_test['flipper_length_mm'].values.reshape(-1,1)
        y_pred = arbol.predict(test_xr)

        st.write('---')
        st.write('**Rango y grafico**')
        
        st.code('''x_vect = np.arange(2750,6350,0.01)   # vector numpy
x_graf = x_vect.reshape((len(x_vect),1))        # dataframe de 1 columna''')
        
        x_vect = np.arange(2750,6350,0.01)   # vector numpy
        x_graf = x_vect.reshape((len(x_vect),1))        # dataframe de 1 columna
        
        # Grafico
        st.code('''fig,ax = plt.subplots() 
ax.scatter(x=test_xr, y=test_yr, color='#ff8c00')   # grafica los datos
ax.plot(x_graf, arbol.predict(x_graf), color='lime')  # id predicion
ax.scatter(test_xr, y_pred, color='red', marker='*')
ax.set_title('Regresion Arbol')
ax.set_xlabel('x')
ax.set_ylabel('y')
                
st.pyplot(fig)''')
        
        
        with st.container(width=800):
            fig,ax = plt.subplots()   
            ax.scatter(x=test_xr, y=test_yr, color='#ff8c00')   # grafica los datos
            ax.plot(x_graf, arbol.predict(x_graf), color='lime')  # id predicion
            ax.scatter(test_xr, y_pred, color='red', marker='*')
          
            ax.set_title('Regresion Arbol')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            st.pyplot(fig)


        st.write('---')
        st.write('###### Arbol como Regre Clasificador')

        st.code('arbolClas = DecisionTreeClassifier(random_state=36)')
        arbolClas = DecisionTreeClassifier(random_state=36)

        st.write('**Entrenamiento**')

        st.code('arbolClas.fit(X_train, y_train)')
        arbolClas.fit(X_train, y_train)

        st.code('''with st.container(width=800):
graf = sns.barplot(                
x = arbolClas.feature_importances_,  
y = X_train.columns,  
palette = 'iridis'            
)

st.pyplot(graf)''')
        
        with st.container(width=800):
            fig,ax = plt.subplots()
            sns.barplot(
            x = arbolClas.feature_importances_,
            y = X_train.columns,
            palette = 'viridis')
            
            st.pyplot(fig)
        
        st.write('---')
        
        st.code('''lista_car = list(x.columns)
fig,ax = plt.subplots()
plot_tree(arbolClas, filled=True, ronded=True, feature_names=lista_car)

st.pyplot(fig)''')
        
        lista_car = list(X.columns)
        fig,ax = plt.subplots()
        plot_tree(arbolClas, filled=True, rounded=True, feature_names=lista_car)
        
        st.pyplot(fig)
         
        st.write('---')
        st.write('**Preccion**')
        
        st.code('y_pred = arbolClas.predict(X_test)')
        y_pred = arbolClas.predict(X_test)
        
        st.code('''st.write(f'Score train: {arbolClas.score(X_train, y_train)}')
st.write(f'Score test: {arbolClas.score(X_test, y_test)}')''')
        
        st.write(f'Score train: {arbolClas.score(X_train, y_train)}')
        st.write(f'Score test: {arbolClas.score(X_test, y_test)}')
        
        st.write('---')
        st.write('**Matriz de Confusion**')
        
        st.code('''from sklearn.metrics import confusion_matrix
with st.container(width=250): 
    st.write(confusion_matrix(y_test, y_pred))''')


        from sklearn.metrics import confusion_matrix
        with st.container(width=250):      
            st.write(confusion_matrix(y_test, y_pred))









    
      
    if opcion_seleccionada == 'Prestamos':
        st.write('##### Data Frame con los datos de Prestatarios')  
        st.write('Se intenta predecir si el prestamo fue devuelvo en su totalidad o no mediante la columna not.fully.paid del Data Frame.')
        st.code('''# Carga de csv con los datos de Prestamos
df_prestamos = pd.read_csv('DataFrames/df_prestamos.csv')     
df_prestamos.head(10)       # head''')  

        df_prestamos = load_date_prestamos()
        st.dataframe(df_prestamos.head(10))
    
        st.code('''# info
df_prestamos.info()''') 
        
        df_prestamos.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')          
        
        st.write('---')
        st.write('#### EDA')

        with st.container(border=True):
            col1, col2 = st.columns(2)
            
            with col1:
            
                st.write('##### Histograma de fico vs credit.policy')
                
                st.code('''fig,ax = plt.subplots()
sns.histplot(df_prestamos, x='fico', hue='credit.policy', bins=35)

st.pyplot(fig)''')
                
                fig,ax = plt.subplots()
                sns.histplot(df_prestamos, x='fico', hue='credit.policy', bins=35)
                
                st.pyplot(fig)

            with col2:
            
                st.write('##### Histograma de fico vs not.fully.paid')
               
                st.code('''fig,ax = plt.subplots()
sns.histplot(df_prestamos, x='fico', hue='not.fully.paid', bins=35)

st.pyplot(fig)''')               
               
                
                fig,ax = plt.subplots()
                sns.histplot(df_prestamos, x='fico', hue='not.fully.paid', bins=35)
                
                st.pyplot(fig)


        with st.container(border=True):
            col1, col2 = st.columns(2)
            
            with col1:
            
                st.write('##### Countplot de purpose por not.fully.paid')
                
                st.code('''fig,ax = plt.subplots()
sns.countplot(data=df_prestamos, x='purpose', hue='not.fully.paid')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

st.pyplot(fig)''')                     
                
                fig,ax = plt.subplots()
               
                sns.countplot(data=df_prestamos, x='purpose', hue='not.fully.paid')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                
                st.pyplot(fig)

            with col2:
                
                st.write('##### Joinplot fico vs interest rate')

                st.code('''graf = sns.jointplot(data=df_prestamos, x='fico',y='int.rate', kind='reg')
                        
st.pyplot(graf)     



''')      

                graf = sns.jointplot(data=df_prestamos, x='fico',y='int.rate', kind='reg')
                st.pyplot(graf)


        with st.container(border=True):
            st.write('##### Lmplot de fico vs interest rate')
            
            fig = sns.lmplot(data=df_prestamos, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set1')
            st.pyplot(fig)


        st.write('---')
        st.write('##### Conversión de variables categóricas')
        st.write('''**get_dummies**: Función para la preparación de datos que convierte variables categóricas (texto) en un formato numérico binario (0 y 1) mediante la codificación One-Hot,  
    creando una nueva columna por cada categoríca única y asignando 1 si la categoría existe en esa fila y 0 si no.''')
        
        st.write('* columns: para especificar que columnas del dataframe se quieren transformar.')
        st.write('* drop_first: si es True, elimina la primera columna generada para cada variable original, ayudando a evitar la multicolinealidad en modelos estadisticos.' )
        
        st.code('''cat_feats = ['purpose']
final_data = pd.get_dummies(data=df_prestamos, columns=cat_feats, drop_first=True)    
final_data.head(10)            
''')
        
        cat_feats = ['purpose']
        
        final_data = pd.get_dummies(data=df_prestamos, columns=cat_feats, drop_first=True)
        st.dataframe(final_data.head(10))


        st.write('---')

        st.write('##### Separación de datos del modelo')
        st.code('''X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']''')

        X = final_data.drop('not.fully.paid', axis=1)
        y = final_data['not.fully.paid']

        st.write('---')
        st.write('##### Entrenamiento del modelo (Decision Tree)')
        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test. El parámetro random_state permite conseguir cierta repetición de los resultados.''')
        st.write('* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)')
        
        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)''')
        


        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

        st.code('''from sklearn.tree import DecisionTreeClassifier
                
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)''')

        from sklearn.tree import DecisionTreeClassifier
                
        dtree = DecisionTreeClassifier()
        dtree.fit(X_train, y_train)

        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado. Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.')
        
        st.code('predictions = dtree.predict(X_test)')
        
        predictions = dtree.predict(X_test)

        st.write('---')
        st.write('##### Reporte de clasificación')

        st.code('''from sklearn.metrics import classification_report 
classification_report(y_test, predictions, output_dict=True)''')
    
        from sklearn.metrics import classification_report      
        st.dataframe(classification_report(y_test, predictions, output_dict=True))

        st.write('---')
        st.write('##### Matrix de Confusión')

        st.code('''from sklearn.metrics import confusion_matrix  
confusion_matrix(y_test, predictions)''')

        from sklearn.metrics import confusion_matrix
        with st.container(width=200):      
            st.write(confusion_matrix(y_test, predictions))

        st.write('---')
        st.write('##### Entrenamiento del modelo (Random Forest)')
        
        st.code('''from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)''')
        

        from sklearn.ensemble import RandomForestClassifier
                
        rfc = RandomForestClassifier(n_estimators=300)
        rfc.fit(X_train, y_train)

        st.write('---')
        st.write('##### Predicciones del conjunto de test')

        st.code('predictions = rfc.predict(X_test)')
        predictions = rfc.predict(X_test)

        st.write('---')
        st.write('##### Reporte de clasificación')      
        
        st.code('classification_report(y_test, predictions, output_dict=True)')
        st.dataframe(classification_report(y_test, predictions, output_dict=True))  

        st.write('##### Matrix de Confusión')   
        
        with st.container(width=200):      
            st.write(confusion_matrix(y_test, predictions))

    if opcion_seleccionada == 'Kyphosis':
        st.write('##### Data Frame con los datos de Cirugias de columna.')
    
        st.code('''# Carga de csv con los datos de Kyphosis
df_kyphosis = pd.read_csv('DataFrames/df_kyphosis.csv')     
df_kyphosis.head(10)       # head''')  

        df_kyphosis = load_data_trees()
        st.dataframe(df_kyphosis.head(10))
    
        st.code('''# info
df_kyphosis.info()''') 
        
        df_kyphosis.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')    
    
        st.write('---')
        st.write('##### Seperación de los datos del modelo')  

        st.code('''X = df_kyphosis.drop('Kyphosis', axis=1)
y = df_kyphosis['Kyphosis']''')

        X = df_kyphosis.drop('Kyphosis', axis=1)
        y = df_kyphosis['Kyphosis']        


        st.write('---')
        st.write('##### Entrenamiento del modelo (Decision Tree)')  

        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.    
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.
El parámetro random_state permite conseguir cierta repetición de los resultados.    
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')
        
        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)''')
        
        st.code('''from sklearn.tree import DecisionTreeClassifier
                
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
''')
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.3, random_state=101)
        
        from sklearn.tree import DecisionTreeClassifier
        
        dtree = DecisionTreeClassifier()
        dtree.fit(X_train, y_train)

        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('''* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado.
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')
        
        st.code('predictions = dtree.predict(X_test)')
        
        predictions = dtree.predict(X_test)
        
        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report
                
st.dataframe(classification_report(y_test, predictions, output_dict=True))''')
            
        from sklearn.metrics import classification_report
        st.dataframe(classification_report(y_test, predictions, output_dict=True))


        st.write('---')
        st.write('##### Matrix de Confusión')    

        st.code('''from sklearn.metrics import confusion_matrix
                
confusion_matrix(y_test, predictions)''')
        
        from sklearn.metrics import confusion_matrix
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))

        st.write('---')
        st.write('##### Entrenamiento del modelo (Random Forest)')  

        st.code('''from sklearn.ensemble import RandomForestClassifier
                
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)''')

        from sklearn.ensemble import RandomForestClassifier

        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train, y_train)

        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        
        st.code('predictions = rfc.predict(X_test)')
        
        predictions = rfc.predict(X_test)

        st.write('---')
        st.write('##### Reporte de clasificación')

        st.code('st.dataframe(classification_report(y_test, predictions, output_dict=True))')
        st.dataframe(classification_report(y_test, predictions, output_dict=True))

        st.write('---')
        st.write('##### Matrix de Confusión')  

        st.code('confusion_matrix(y_test, predictions)')
        
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))



def ml_knn():
    buffer = io.StringIO()
    
    st.write('#### Definición')
    st.write('''Es un algoritmo de aprendizaje automático superviasado que se utiliza para tareas de clasificación y regresión.     
El algoritmo funciona encontrando los "k" puntos de datos más próximos a un nuevo punto de datos y asignando a este nuevo punto la etiqueta más común entre sus vecinos (para clasificación)        
o el promedio de sus valores (para regresión).''')
    st.write('---')    
    
    opciones_mlknn = ['Classified Data','Iris','Ejemplo Regresion Aleatorio']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlknn)
        st.success(f'##### **{opcion_seleccionada}** ')


        @st.cache_data
        def load_data_classified_data():
            df_classified = pd.read_csv('DataFrames/Classified Data', index_col=0)
            
            return df_classified


    if opcion_seleccionada == 'Ejemplo Regresion Aleatorio':
        st.write('##### Generación de un arreglo con datos aleatorios.')

        st.code('''import numpy as np
from sklearn import neighbors''')
        
        import numpy as np
        from sklearn import neighbors
        
        st.code('''np.random.seed(0)
X = np.sort(5 * np.random.rand(40,1), axis=0)''')
        
        np.random.seed(0)
        X = np.sort(5 * np.random.rand(40,1), axis=0)
        
        st.code(X, language='html')
        
        st.code('y = np.sin(X).ravel()')
        y = np.sin(X).ravel()
        
        st.code(y, language='html')

        #Agregar ruido al target
        st.write('Agregar ruido al target')
        st.code('y[::5] += 1 * (0.5 - np.random.rand(8))')
        y[::5] += 1 * (0.5 - np.random.rand(8))
        
        st.code(y, language='html')

        st.write('---')
        
        
        st.code('''n_neighbors = 5
T = np.linspace(0,5,500)[:, np.newaxis]''')
        
        n_neighbors = 5
        T = np.linspace(0,5,500)[:, np.newaxis]
        
        #for i, weights in enumerate(['uniform','distance']):
        
        st.code('''knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
y_ = knn.fit(X,y).predict(T)''')
        
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
        y_ = knn.fit(X,y).predict(T)
            
            
        st.code('''fig,ax = plt.subplots()
    plt.scatter(X,y,color='darkorange', label='data')
    plt.plot(T,y_,color='navy',label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title(f'KNeighborsRegressor (K = {n_neighbors}, weights=uniform)')
                    
    plt.tight_layout()

    st.pyplot(fig)''')    
            
        with st.container(width=1300):    
            
            fig,ax = plt.subplots()
            plt.scatter(X,y,color='darkorange', label='data')
            plt.plot(T,y_,color='navy',label='prediction')
            plt.axis('tight')
            plt.legend()
            plt.title(f'KNeighborsRegressor (K = {n_neighbors}, weights=uniform)')
                
            plt.tight_layout()
            st.pyplot(fig)


        st.code('''knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_ = knn.fit(X,y).predict(T)''')
        
        
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
        y_ = knn.fit(X,y).predict(T)
            
        st.code('''fig,ax = plt.subplots()
    plt.scatter(X,y,color='darkorange', label='data')
    plt.plot(T,y_,color='navy',label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title(f'KNeighborsRegressor (K = {n_neighbors}, weights=distance)')
                    
    plt.tight_layout()

    st.pyplot(fig)''')                
            
            
        with st.container(width=1300):    
            
            fig,ax = plt.subplots()
            plt.scatter(X,y,color='darkorange', label='data')
            plt.plot(T,y_,color='navy',label='prediction')
            plt.axis('tight')
            plt.legend()
            plt.title(f'KNeighborsRegressor (K = {n_neighbors}, weights=distance)')
                
            plt.tight_layout()
            st.pyplot(fig)





    if opcion_seleccionada == 'Iris':
        st.write('##### Data Frame con los datos de Flores.')

        st.code('df_iris = pd.read_csv\'DataFrames/Iris.csv\',index_col=\'Id\')')
        df_iris = pd.read_csv('DataFrames/Iris.csv',index_col='Id')
        
        st.dataframe(df_iris.head(10))
        
        st.code('''# info
df_iris.info()''') 
        
        df_iris.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')    

        
        st.write('---')
        st.write('**Conversion de la categoria Species en numeros**')
        
        st.code('df_iris[\'Species\').unique()')
        with st.container(width=200):
            
            st.write(df_iris['Species'].unique())
        
        
        st.code('''data = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}
        
df_iris['Species'] = df_iris['Species'].map(data)''')
        
        data = {
            'Iris-setosa':0,
            'Iris-versicolor':1,
            'Iris-virginica':2
        }
        
        df_iris['Species'] = df_iris['Species'].map(data)
        
        st.dataframe(df_iris.head(10))
        
        st.write('---')
        st.write('##### Seperación de los datos del modelo')
        
        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.    
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.''')
        
        st.code('''X = df_iris.drop('Species',axis=1)  
y = df_iris['Species']''')
        
        X = df_iris.drop('Species',axis=1)        # Eliminar columna Species
        y = df_iris['Species']

        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)''')

        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.3, random_state=1, stratify=y)


        st.write('---')
        st.write('##### Escalamiento de datos')    

        st.write('''* **StandardScaler**: es una herrmaienta de preprocesamiento de datos que se utiliza para estandarizar funciones eliminando la media y escalando a la varianza media.     
Muchos algoritmos de ML funcionan mejor o convergen más rápido cuando las funciones están en una escala similar y centradas alrededor de cero.       
StandardScaler aborda esto transformando los datos de modo que cada característica tenga una media de 0 y una desviación estándar de 1.
* **fit(data)**: se utiliza para calcular la media y la desviación estándar de una característica determinada que se utilizará posteriormente para escalar.
* **transform(data)**: se utiliza para realizar el escalamiento utilizando la media y la desviación estándar calculadas utilizando el método .fit()
''')
        
        st.code('''from sklearn.preprocessing import StandardScaler
                
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
''')
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
           
        scaler.fit(X_train)
        X_train_std = scaler.transform(X_train)
        X_test_std = scaler.transform(X_test)


        st.write('---')
        st.write('##### Entrenamiento del modelo')

        st.write('''* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')
        
        st.code('''from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski',weights='uniform')
knn.fit(X_train, y_train)
''')
        
        from sklearn.neighbors import KNeighborsClassifier
    
        knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski',weights='uniform')
        knn.fit(X_train_std, y_train)

        
        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('''* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado.
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')
        
        st.code('predictions = knn.predict(X_test_std)')
        
        
        predictions = knn.predict(X_test_std)

        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report
                
st.dataframe(classification_report(y_test, predictions output_dict=True))''')
            
        from sklearn.metrics import classification_report
        st.dataframe(classification_report(y_test, predictions, output_dict=True))


        st.write('---')
        st.write('##### Matrix de Confusión')    

        st.code('''from sklearn.metrics import confusion_matrix
                
confusion_matrix(y_test, predictions)''')
        
        from sklearn.metrics import confusion_matrix
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))

        st.write('---')
        st.write('##### Tasa de Error') 
        
        st.code('''error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_std, y_train)
            
    pred_i = knn.predict(X_test_std)
    error_rate.append(np.mean(pred_i != y_test))                     # Tasa de error promedio
    
    
fig, ax = plt.subplots()
ax.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
            
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel("Error Rate")
            
st.pyplot(fig)    
''')
    
        error_rate = []
        for i in range(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train_std, y_train)
            
            pred_i = knn.predict(X_test_std)
            error_rate.append(np.mean(pred_i != y_test))                     # Tasa de error promedio
         
        with st.container(border=True, width=1000):        
            fig, ax = plt.subplots()
            ax.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
            
            plt.title('Error Rate vs K Value')
            plt.xlabel('K')
            plt.ylabel("Error Rate")
            
            st.pyplot(fig)

        st.write('---')
        st.write('**Entrenamiento del modelo con k=4**')   
        
        st.code('''knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski',weights='uniform')
knn.fit(X_train_std, y_train)
predictions = knn.predict(X_test_std)
        
st.dataframe(classification_report(y_test, predictions, output_dict=True))

confusion_matrix(y_test, predictions)''')
        
        
        knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski',weights='uniform')
        knn.fit(X_train_std, y_train)
        predictions = knn.predict(X_test_std)

        
        st.write('**Reporte de Clasificacion**')
        st.dataframe(classification_report(y_test, predictions, output_dict=True))
        
        st.write('**Matriz de Confusion**')
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))     









    if opcion_seleccionada == 'Classified Data':
        st.write('##### Data Frame con los datos Clasificados de Personal.')
    
        st.code('''# Carga de csv con los datos de Classified Data
df_classified = pd.read_csv('DataFrames/Classified Data', index_col=0)     
df_classfied.head(10)       # head''')  

        df_classified = load_data_classified_data()
        st.dataframe(df_classified.head(10))
    
        st.code('''# info
df_classfied.info()''') 
        
        df_classified.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')    
    
        st.write('---')
        st.write('##### Escalamiento de datos')    

        st.write('''* **StandardScaler**: es una herrmaienta de preprocesamiento de datos que se utiliza para estandarizar funciones eliminando la media y escalando a la varianza media.     
Muchos algoritmos de ML funcionan mejor o convergen más rápido cuando las funciones están en una escala similar y centradas alrededor de cero.       
StandardScaler aborda esto transformando los datos de modo que cada característica tenga una media de 0 y una desviación estándar de 1.
* **fit(data)**: se utiliza para calcular la media y la desviación estándar de una característica determinada que se utilizará posteriormente para escalar.
* **transform(data)**: se utiliza para realizar el escalamiento utilizando la media y la desviación estándar calculadas utilizando el método .fit()
''')
        
        st.code('''from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_classified_data = df_classified.drop('TARGET CLASS', axis=1)   # Se elimina la columna TARGET CLASS
scaler.fit(df_classified_data)
scaled_features = scaler.transform(df_classified_data)     
           
df_feat = pd.DataFrame(scaled_features, columns=df_classified_data.columns)     # Data Frame con los datos escalados.
st.dataframe(df_feat.head(10))''')
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Eliminar ultima columna
        df_classified_data = df_classified.drop('TARGET CLASS', axis=1)    
        
        scaler.fit(df_classified_data)
        scaled_features = scaler.transform(df_classified_data)
        
        df_feat = pd.DataFrame(scaled_features, columns=df_classified_data.columns)
    
        
        st.dataframe(df_feat.head(10))

        st.write('---')
        st.write('##### Seperación de los datos del modelo')
        
        st.code('''X = df_feat
y = df_classified['TARGET CLASS']''')
        
        X = df_feat
        y = df_classified['TARGET CLASS']
        
        st.write('---')
        st.write('##### Entrenamiento del modelo')

        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.    
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.
El parámetro random_state permite conseguir cierta repetición de los resultados.    
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')
        
        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)''')
        
        st.code('''from sklearn.neighbors import KNeighborsClassifier
                
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
''')
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.3, random_state=101)
        
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('''* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado.
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')
        
        st.code('predictions = knn.predict(X_test)')
        
        predictions = knn.predict(X_test)
        
        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report
                
target = ['Target Class = 0','Target Class = 1']
st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))''')
            
        target = ['Clicked on ad = 0','Clicked on ad = 1']
        from sklearn.metrics import classification_report
        st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))

    
        st.write('---')
        st.write('##### Matrix de Confusión')    

        st.code('''from sklearn.metrics import confusion_matrix
                
confusion_matrix(y_test, predictions)''')
        
        from sklearn.metrics import confusion_matrix
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))
        
        st.write('---')
        st.write('##### Tasa de Error') 
        
        st.code('''error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
            
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))                     # Tasa de error promedio
    
    
fig, ax = plt.subplots()
ax.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
            
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel("Error Rate")
            
st.pyplot(fig)    
''')
    
        error_rate = []
        for i in range(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))                     # Tasa de error promedio
         
        with st.container(border=True, width=1000):        
            fig, ax = plt.subplots()
            ax.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
            
            plt.title('Error Rate vs K Value')
            plt.xlabel('K')
            plt.ylabel("Error Rate")
            
            st.pyplot(fig)
        
        st.write('---')
        st.write('**Entrenamiento del modelo con k=17**')   
        
        st.code('''knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
        
    st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))

    confusion_matrix(y_test, predictions) 
    ''')
        
        knn = KNeighborsClassifier(n_neighbors=17)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
    
        st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))       
        
        
def ml_regresion_logistica():
    
    buffer = io.StringIO()
    
    st.write('#### Definición')
    st.write('''La regresión logística es un método estadístico para predecir la probabilidad de un resultado categórico, como \'si\' o \'no\'.     
Utiliza una función logística (curva sigmoidea) para modelar la relación entre las variables independientes y una variable dependiente binaria (que solo tiene dos resultados).
Este modelo es útil en campos como la medicina para predecir la probabilidad de enfermedades o en la economía para predecir resultados de acciones.''')
    st.write('---')
    
    opciones_mlreglog = ['titanic','advertising']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlreglog)
        st.success(f'##### **{opcion_seleccionada}** ')
        
        @st.cache_data
        def load_data_titanic():
            # Carga del dataframe
            df_train = pd.read_csv('DataFrames/titanic_train.csv')
           # df_test = pd.read_csv('DataFrames/titanic_test.csv')

            return df_train  #, df_test
        
        @st.cache_data
        def load_data_advertising():
            df_advertising = pd.read_csv('DataFrames/advertising.csv')
            
            return df_advertising
        
    if opcion_seleccionada == 'advertising':
        st.write('##### Data Frame con los datos de clics en anuncios de internet por parte de los usuarios.')
    
        st.code('''# Carga de csv con los datos de advertising
df_advertising = pd.read_csv('DataFrames/advertising.csv')      
df_advertising.head(10)       # head''')  

        df_advertising = load_data_advertising()
        st.dataframe(df_advertising.head(10))
    
        st.code('''# info
df_advertising.info()''') 
        
        df_advertising.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')    
    
        st.write('---')
        st.write('#### EDA')    
    
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write('###### Histograma Age')   
                st.code('''fig, ax = plt.subplots()
ax.hist(x=df_advertising['Age'], bins=20, edgecolor='#000000', 
color='#8b92cc', alpha=.8)
ax.set_xlabel('Edad')
ax.set_ylabel('Frecuencia')

st.pyplot(fig)''')

                fig, ax = plt.subplots()
                ax.hist(x=df_advertising['Age'], bins=30, edgecolor='#000000', color='#8b92cc', alpha=.8)
                ax.set_title('Histograma Edades Usuarios')
                ax.set_xlabel('Edad')
                ax.set_ylabel('Frecuencia')

                st.pyplot(fig)
                
                
            with col2:
                st.write('###### Joinplot Area Income vs Age')
                st.code('''graf = sns.jointplot(data=df_advertising, x='Age',y='Area Income', kind='reg')
                        
st.pyplot(graf)





''')
                graf = sns.jointplot(data=df_advertising, x='Age',y='Area Income', kind='reg')

                st.pyplot(graf)
    
    
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write('###### Joinplot mostrando distribución kde de Daily Time spent on site vs. Age.')   
                st.code('''graf = sns.jointplot(data=df_advertising, x='Age', y='Daily Time Spent on Site', 
kind='kde', color='red') 
st.pyplot(graf)''')
                    
                                       
                graf = sns.jointplot(data=df_advertising, x='Age', y='Daily Time Spent on Site', kind='kde', color='red')                   
                st.pyplot(graf)

            with col2:
                st.write('###### Joinplot de Daily Time Spent on Site vs. Daily Internet Usage')                        
                st.code('''graf = sns.jointplot(data=df_advertising, x='Daily Time Spent on Site', y='Daily Internet Usage', kind='scatter')   
                        
st.pyplot(graf) ''')
                
                
                graf = sns.jointplot(data=df_advertising, x='Daily Time Spent on Site', y='Daily Internet Usage', kind='scatter')                   
                st.pyplot(graf)                    
                    
        with st.container(border=True):
            st.write('##### Pairplot sobre Clicked on Ad')      
            st.code('''graf = sns.pairplot(data=df_advertising, hue='Clicked on Ad') 
                    
st.pyplot(graf)''')
            graf = sns.pairplot(data=df_advertising, hue='Clicked on Ad')      
            st.pyplot(graf)
    
    

        st.write('---')
        st.write('##### Separación de los datos del Modelo')       
        
        st.code('''X = df_advertising.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis=1)
y = df_advertising['Clicked on Ad']   ''')    
        
        X = df_advertising.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis=1)
        y = df_advertising['Clicked on Ad']     
            
        
        st.write('---')
        st.write('##### Entrenamiento del Modelo')  

        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.    
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.
El parámetro random_state permite conseguir cierta repetición de los resultados.    
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')
        
        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)''')
        
        st.code('''from sklearn.linear_model import LogisticRegression
                
lg = LogisticRegression()
lg.fit(X_train, y_train)
''')
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        from sklearn.linear_model import LogisticRegression
        lg = LogisticRegression()
        lg.fit(X_train, y_train)
    
    
        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('''* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado.
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')

        st.code('''predictions = lg.predict(X_test)''')
        predictions = lg.predict(X_test)
    
    
        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report
                
target = ['Clicked on ad = 0','Clicked on ad = 1']
st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))''')
            
        target = ['Clicked on ad = 0','Clicked on ad = 1']
        from sklearn.metrics import classification_report
        st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))

    
        st.write('---')
        st.write('##### Matrix de Confusión')    

        st.code('''from sklearn.metrics import confusion_matrix
                
confusion_matrix(y_test, predictions)''')
        
        from sklearn.metrics import confusion_matrix
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))
    
    if opcion_seleccionada == 'titanic':
        st.write('##### Data Frame con datos del naufragio del Titanic')                                        

        st.code('''# Carga de csv con los datos de train
df_train = pd.read_csv('DataFrames/titanic_train.csv')      
df_train.head(10)       # head''')
        
        #df_test = pd.read_csv('DataFrames/titanic_test.csv')
        
       # df_train, df_test = load_data_titanic()    
        df_train = load_data_titanic()
        st.dataframe(df_train.head(10))       # head
        st.code('''# info
df_train.info()''') 
        
        df_train.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')

        st.write('---')
        st.write('#### EDA')

        st.write('##### Conteo de Sobrevivientes')
        st.write('''0 : No sobrevivió   
1 : Sobrevivió.''')     
        
        st.code('df_train[\'Survived\'].value_counts()')
        st.dataframe(df_train['Survived'].value_counts(), width=300)

        
        col1, col2 = st.columns(2)
        
        with col1:
            # sobrevivientes Male
            st.write('Sobrevivientes Hombres')
            st.code('''filtro = df_train['Sex'] == 'male'
df_train[filtro]['Survived'].value_counts()''')
            filtro = df_train['Sex'] == 'male'
            st.dataframe(df_train[filtro]['Survived'].value_counts())
        
        with col2:
            # sobrevivientes Female
            st.write('Sobrevivientes Mujeres')
            st.code('''filtro = df_train['Sex'] == 'female'
df_train[filtro]['Survived'].value_counts()''')            
            filtro = df_train['Sex'] == 'female'
            st.dataframe(df_train[filtro]['Survived'].value_counts())        
               

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write('###### Grafico de conteo de Sobrevivientes.')
                st.code('''fig,ax = plt.subplots()
sns.set_style('darkgrid')   
sns.countplot(data=df_train, x='Survived')            
st.pyplot(fig)''')
        
                fig,ax = plt.subplots()
                sns.set_style('darkgrid')
                sns.countplot(data=df_train, x='Survived')
                st.pyplot(fig)

            with col2:
                st.write('###### Grafico de conteo de Sobrevivientes por Sexo.')
                st.code('''fig,ax = plt.subplots()
sns.set_style('darkgrid')   
sns.countplot(data=df_train, x='Survived',hue='Sex')         
st.pyplot(fig)''')
                
                fig,ax = plt.subplots()
                sns.set_style('darkgrid')
                sns.countplot(data=df_train, x='Survived',hue='Sex')
                st.pyplot(fig)
  

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write('###### Grafico de conteo de Sobrevivientes por Clase.')
                st.code('''fig,ax = plt.subplots()
sns.set_style('darkgrid')   
sns.countplot(data=df_train, x='Survived')            
st.pyplot(fig)''')
                fig,ax = plt.subplots()
                sns.set_style('darkgrid')
                sns.countplot(data=df_train, x='Survived',hue='Pclass')
                st.pyplot(fig)
                
            with col2:
                st.write('###### Grafico de distribución de Pasajeros por Edad.' )
                st.code('''fig,ax = plt.subplots()
sns.set_style('darkgrid')   
fig = sns.displot(data=df_train, x='Age',bins=30,kde=True)          
st.pyplot(fig)''')
                fig,ax = plt.subplots()
                sns.set_style('darkgrid')
                fig = sns.displot(data=df_train, x='Age',bins=30,kde=True)
                st.pyplot(fig)

        st.write('---')

        st.write('##### Reemplazo de datos nulos.')
        st.write('''Existen datos nulos en Age, por lo tanto se puede reemplazar los valores nulos por el promedio de edad.     
Se calcula el promedio de edad por clase.''')
        st.code('''df_train.isnull().sum()''')
        
        #col1, col2 = st.columns(2)
        
        with st.container(width=200):
            #with col1:
            st.dataframe(df_train.isnull().sum())
           # with col2:
            #    st.dataframe(df_test.isnull().sum())
        
        median_train_1 = df_train[df_train['Pclass'] == 1]['Age'].median()      # edad promedio 1era Clase
        median_train_2 = df_train[df_train['Pclass'] == 2]['Age'].median()      # edad promedio 2da Clase
        median_train_3 = df_train[df_train['Pclass'] == 3]['Age'].median()      # edad promedio 3era Calse
        
        #median_test = df_test['Age'].median()
        
        st.write('Cálculo de la media de Edad para train y test')
        st.code('''median_train_1 = df_train[df_train['Pclass'] == 1]['Age'].median()   
median_train_2 = df_train[df_train['Pclass'] == 2]['Age'].median()  
median_train_3 = df_train[df_train['Pclass'] == 3]['Age'].median()''')
        
        st.code(f'''Media de edad df_train:     
1era clase: {median_train_1}        
2da clase: {median_train_2}
3era clase: {median_train_3}''')
       

        
        st.write('''##### apply()
Es una herramienta de propósito general para aplicar una función a lo largo de un eje (filas o columnas) de un DataFrame o a cada elemento de una Serie.    
Se utiliza para realizar transformaciones, cálculos y lógica condicional compleja, siendo una alternativa más eficiente y limpia que los bucles.    
Para usarlo, se le pasa la función a aplicar, y el parámetro axis determina si se aplica a las columnas (\(0\)) o a las filas (\(1\)
''')
        
        
        def categorizar(fila):
            if pd.isnull(fila['Age']):          # Valor de la fila en el campo Age es nulo
                if fila['Pclass'] == 1:
                    return median_train_1
                elif fila['Pclass'] == 2:
                    return median_train_2
                elif fila ['Pclass'] == 3:
                    return median_train_3
            else:
                return fila['Age']
        
        
        df_train['Age'] = df_train.apply(categorizar, axis=1)
        
        st.code('''def categorizar(fila):
    if pd.isnull(fila['Age']):      # Valor de la fila en el campo Age es nulo
        if fila['Pclass'] == 1:
            return median_train_1
        elif fila['Pclass'] == 2:
            return median_train_2
        elif fila ['Pclass'] == 3:
            return median_train_3
    else:
        return fila['Age']  
        
df_train['Age'] = df_train.apply(categorizar, axis=1)       # axis = 1 (se aplica sobre las filas)
''')

        st.write('---')
        st.write('##### Eliminación de columnas.')
        st.write('''Existe muchos datos nulos en la columna Cabin, por lo tanto es conveniente eliminar la columna del DataFrame.        
También se eliminan las columnas Passenger Id, Name y Ticket ya que no son relevantes.''')
        st.code('''df_train.drop(['Cabin','PassengerId,''Name','Ticket'], axis=1, inplace=True)''')     

        df_train.drop(['Cabin','PassengerId','Name','Ticket'], axis=1, inplace=True)

    
        st.write('---')
        st.write('##### Eliminación de filas.')
        st.write('''Existes dos filas con valor nulo en la columna Embarked. Se eliminan los 2 registros.''')
        st.code('''df_train.dropna(axis=0, inplace=True)''')     
        
        df_train.dropna(axis=0, inplace=True)
        
  
        st.write('---')
        st.write('##### Reemplazo de Variables categorícas.')
        st.write('''Es necesario reemplazar las variables categóricas (Sex, Embarked) por valores numéricos     
Sex = 0 (female), 1 (male)  
Embarked = S (1), C (2), Q (3)
''')

        def mod_sexo(sex):
            if sex == 'female':
                return 0
            elif sex == 'male':
                return 1
            
        def mod_ciudad(ciudad):
            if ciudad == 'S':
                return 1
            elif ciudad == 'C':    
                return 2
            elif ciudad == 'Q':
                return 3
            
        df_train['Sex'] = df_train['Sex'].map(mod_sexo)
        df_train['Embarked'] = df_train['Embarked'].map(mod_ciudad)
        
        st.code('''def mod_sexo(sex):
    if sex == 'female':
        return 0
    elif sex == 'male':
        return 1

def mod_ciudad(ciudad):
     if ciudad == 'S':
        return 1
    elif ciudad == 'C':    
        return 2
    elif ciudad == 'Q':
        return 3        
         
df_train['Sex'] = df_train['Sex'].map(mod_sexo)
df_train['Embarked'] = df_train['Embarked'].map(mod_ciudad)''')
        
  
        st.write(f'Cantiad de filas/columnas: {df_train.shape}')
        st.dataframe(df_train.head(10))
 
        st.write('---')
        st.write('##### Separación de los datos del Modelo') 

        st.code('''X = df_train.drop('Survived', axis=1)
y = df_train['Survived']''')
        
        X = df_train.drop('Survived', axis=1)
        y = df_train['Survived']
    
        st.write('---')
        st.write('##### Entrenamiento del Modelo')
        
        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.    
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.
El parámetro random_state permite conseguir cierta repetición de los resultados.    
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')
        
        st.code('''from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)''')
        
        st.code('''from sklearn.linear_model import LogisticRegression
                
lg = LogisticRegression()
lg.fit(X_train, y_train)
''')
        
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        
        from sklearn.linear_model import LogisticRegression
        
        lg = LogisticRegression()
        lg.fit(X_train, y_train)
        
        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('''* **predict**:  función que se utiliza para obtener predicciones de un modelo entrenado.    
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')
        
        st.code('predictions = lg.predict(X_test)')
        
        predictions = lg.predict(X_test)

        st.write('---')
        st.write('##### Reporte de clasificación')
        
        st.code('''from sklearn.metrics import classification_report
                
target = ['Survived = 0','Survived = 1']
st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))''')
        
        from sklearn.metrics import classification_report
        
        target = ['Survived = 0','Survived = 1']
        st.dataframe(classification_report(y_test, predictions, target_names=target, output_dict=True))

        st.write('---')
        st.write('##### Matrix de confusión')

        st.code('''from sklearn.metrics import confusion_matrix
                
confusion_matrix(y_test, predictions)''')
        
        from sklearn.metrics import confusion_matrix
        with st.container(width=300):
            st.write(confusion_matrix(y_test, predictions))

       
        
        
        
        
def ml_regresion_lineal():
    
    buffer = io.StringIO()
    
    st.write('#### Definición')
    st.write('''
La regresión lineal es un método estadístico que trata de modelar la relación entre una variable continua y una o más variables independientes mediante el ajuste de una ecuación lineal.   
Se llama regresión lineal simple cuando solo hay una variable independiente y regresión lineal múltiple cuando hay más de una.  
Dependiendo del contexto, a la variable modelada se le conoce como variable dependiente o variable respuesta, y a las variables independientes como regresores, predictores o features.         

**Regresion lineal simple**     

El objetivo de este regresion es predecir el valor de una variable dependiente a partir de una variable independiente.     
Cuanto mayor sea la relacion lineal entre la variable independiente y la variable dependiente, mas precisa sera la prediccion.  
Esto va unido al hecho de que cuanto mayor sea la proporcion de la varianza de la variable dependiente que pueda explicar la variable independiente, mas exacta sera la prediccion.     
La tarea de la regresion lineal simple consiste en determinar exactamente la linea recta que mejor describe la relacion lineal entre la variable dependiente y la independiente.        
Para determinar esta linea recta, se utiliza el metodo de los minimos cuadrados.''')
    
    st.write('**Parametros principales**')
    st.write('''* fit_intercept: por defecto es True. Determina si se debe calcular la intereseccion para este modelo. Si es False, no se calculara (los datos se asumen centrados).
* copy_x: por defecto es True. Si es True, se copia X; de lo contrario, puede ser sobrescrito.
* n_jobs: por defecto None. Numero de trabajos a utilizar para el calcuilo. -1 significa usar todos los procesadores.
''')

    st.write('**Atributos principales**')
    st.write('''* coef_: determina la pendiente de la recta.
* intercept_: lugar de intercepcion de la recta con el eje x.
''')

    
    st.divider()
    
    st.write('##### Tipos')
    st.write('**Regresion Ridge**')
    st.write('''Tambien conocida como regularizacion L2, es un metodo estadistico para reducir errores causados por el sobreajuste de los datos de entrenamiento. La regresion corrige especificamente
la multicolinealidad en el analisis de regresion. Esto resulta util cuando se desarrollan modelos de machine learning que tienen un gran numero de parametros, sobre todo si esos parametros
tambien tienen pesos elevados. ''')
    
    st.write('**Regresion Lasso**')
    st.write('''Es una tecnica de regularizacion que aplica una penalizacion para evitar el sobreajuste y mejorar la precision de los modelos estadisticos. Tambien es conocida como regularizacion L1, 
es una forma de regularizacion para modelos de regresion linel, es un metodo estadistico para reducir los errores causados por el sobreajuste de los datos de entrenamiento.''')
    
    st.write('**Elasticnet**')
    st.write('''Es un algoritmo de regresion linea regularizada que combina penalizacione L1 (Lasso) y L2 (Ridge). Esta tecnica mejor la precision y gestiona la colinealidad (alta correlacion entre variables)
al contraer coeficientes y establecer algunos a cero, logrando modelos mas dispersos y robustos. Se usa para evitar el sobreajuste cuando hay muchas variables predictoras.''')
    
    st.write('**Descenso de Gradiente**')
    st.write('''Es un algoritmo de optimizacion para encontrar la linea de mejor ajuste (y = mx + b) que obtimiza el error entre los valores predichos y los reales. Utiliza el descenso de gradiente
para ajustar iterativamente los parametros m (pendiente) y b (interseccion) mediante el calculo de derivadas, reduciendo el error cuadratico medio hasta converger al minimo.''')
    
    
    st.divider()
    
    opciones_mlreglin = ['Peso de un material','USA Housing', 'Ecommerce Customers']
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlreglin)
        st.success(f'##### **{opcion_seleccionada}** ')
    
        @st.cache_data
        def load_data_usahouse():
            # Carga del dataframe
            df = pd.read_csv('DataFrames/Casas.csv')
            
            return df
    
    if opcion_seleccionada == 'Peso de un material':
        from sklearn import linear_model

        st.write('###### Data Frame con los pesos y longitud de materiales metalicos.')

        st.code('''# Peso especifico (Unidades de g/cm3 )
# barras de base 1 cm2 y largo 'l' 
# R peso especifico -> hay que estimar 
# se pesan las barras con una balanza con errores (pequeños pero desconocidos)
# barra de largo 'm' -> volumen m cm3
# Peso = R*m, peso se mide con errores, m se conoce

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model''')
            
        # Coeficiente de correlacion de Pearson 
        
        st.write('**Coeficiente de correlacion de Pearson**')
        
        st.write('''Dados, dos np.arrays x e y, calcula el coeficiente de correlacion para un ajuste lineal calculado
por minimos cuadrados. Es el coeficiente de correlacion de Pearson, que indica que tan buena es la relacion
lineal entre dos variables, siendo 1 o -1 correlacion perfecta y 0 nula correlacion)''')
        
        st.code('''def coef_corr(x,y): 
        arriba = sum(((x - x.mean())*(y - y.mean())))
        abajo = sum(((x - x.mean())**2)) * sum(((y - y.mean())**2))

        corr = arriba / np.sqrt(abajo)

        return corr''')
 
            
        def coef_corr(x,y): 


                arriba = sum(((x - x.mean())*(y - y.mean())))
                abajo = sum(((x - x.mean())**2)) * sum(((y - y.mean())**2))

                corr = arriba / np.sqrt(abajo)

                return corr


        # Carga de los datos desde el csv
        df = pd.read_csv('DataFrames/longitudes_y_pesos.csv')
        st.dataframe(df.head(10))       # head

        codigo = '''# info
df.info()'''
        st.code(codigo)        
        
        df.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')
        
        codigo = '''# describre
df.describre()'''
        st.code(codigo)        
        
        st.write(df.describe())           

        st.divider()

        st.code('''# Separacion de los datos del modelo 
X = df[['longitud']]
y = df['peso'] ''')


        # Separacion de los datos del modelo 
        X = df[['longitud']]
        y = df['peso']    

        st.code('''# Entrenamiento del Modelo
lm = linear_model.LinearRegression(fit_intercept=False)  # no hay ordenada al origen
lm.fit(X, y)''')

        # Entrenamiento del Modelo
        lm = linear_model.LinearRegression(fit_intercept=False)  # no hay ordenada al origen
        lm.fit(X, y)


        st.code('''# coeficiente para la caracteristica (pendiente)
R = lm.coef_  
R''')

        # coeficiente para la caracteristica (pendiente)
        R = lm.coef_  
        st.code(R)
        
        
        st.code('''errores = y - (lm.predict(X))
ecm = (errores**2).mean
ecm''')        
        
        errores = y - (lm.predict(X))
        ecm = (errores**2).mean
        st.code(ecm, language='html')

        st.code('''# Coeficiente de correlacion 
r2 = coef_corr(df['longitud'], df['peso'])
r2''') 

        # Coeficiente de correlacion 
        r2 = coef_corr(df['longitud'], df['peso'])
        st.code(r2, language='html')

        st.divider()
        
        st.code('''minlong = 0   #limite inferior para el ajuste
maxlong = 30  #limite superior
grilla_longitud = np.linspace(start=minlong, stop=maxlong, num=1000)
grilla_peso = grilla_longitud * R  # recta de ajusta x minimos cuadrados''')
        
        minlong = 0   #limite inferior para el ajuste
        maxlong = 30  #limite superior
        grilla_longitud = np.linspace(start=minlong, stop=maxlong, num=1000)
        grilla_peso = grilla_longitud * R  # recta de ajusta x minimos cuadrados
        
        st.code('''fig,ax = plt.subplots() 
                
plt.scatter(X, y, c='purple', marker='x')

plt.plot(grilla_longitud, grilla_peso, c='magenta')
plt.xlabel('Longitud')
plt.ylabel('Peso')
plt.tight_layout()

st.pyplot(fig)''')
        
        
        with st.container(width=800):
                fig,ax = plt.subplots() 
                
                plt.scatter(X, y, c='purple', marker='x')

                plt.plot(grilla_longitud, grilla_peso, c='magenta')
                plt.xlabel('Longitud')
                plt.ylabel('Peso')
                plt.tight_layout()

                st.pyplot(fig)


    
    # Precios de casa en Estados Unidos
    if opcion_seleccionada == 'USA Housing':
        st.write('###### Data Frame con los precios de casas en Estados Unidos')
        
        codigo = '''# Carga de csv con los datos
df = pd.read_csv('DataFrames/USA_housing.csv')
st.dataframe(df.head(10))       # head'''
        st.code(codigo)
        
        df = load_data_usahouse()    
        st.dataframe(df.head(10))       # head
        codigo = '''# info
st.dataframe(df.info())       # info'''
        st.code(codigo)        
        
        df.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')

        st.write('##### Histograma Precio')
        codigo = '''fig, ax = plt.subplots()    
ax.hist(x=df['Price'], bins=40, edgecolor='#000000')
ax.set_title('Histograma Precio Propiedades')
ax.set_xlabel('Precio')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)'''        
        st.code(codigo)
        
        with st.container(width=800):
            fig, ax = plt.subplots()
            ax.hist(x=df['Price'], bins=40, edgecolor='#000000')
            ax.set_title('Histograma Precio Propiedades')
            ax.set_xlabel('Precio')
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)     
                
        st.write('##### Matriz de Correlación')

        codigo = '''df_numericas = df.select_dtypes(include=['float64'])
matrix_correlacion = df_numericas.corr()     
st.write(matrix_correlacion)'''
        st.code(codigo)

        df_numericas = df.select_dtypes(include=['float64'])
        matrix_correlacion = df_numericas.corr()
        st.write(matrix_correlacion)

        codigo = '''fig, ax = plt.subplots()
sns.heatmap(matrix_correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.3)
ax.set_title('Heatmap')
st.pyplot(fig)'''   
        st.code(codigo)

        with st.container(width=800):
            fig, ax = plt.subplots()
            sns.heatmap(matrix_correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.3)
            ax.set_title('Heatmap')
            st.pyplot(fig)
        
        st.write('---')
        st.write('##### Separacion de los datos del modelo')
        
        codigo = '''X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]  # Datos Independientes
y = df['Price']     # Dato a predecir'''
        st.code(codigo)
        
        X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]  # Datos Indendientes
        y = df['Price']     # Dato a predecir
        st.write('---')
        st.write('##### Entrenamiento del Modelo')
        st.write('''* **train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.     
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.     
El parámetro random_state permite conseguir cierta repetición de los resultados.
* **fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)             
''')
        codigo = '''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

lm = LinearRegression()     # instancia del objeto LinearRegression
lm.fit(X_train, y_train)    # entrenamiento del modelo
lm.intercept_               # punto de intreseccion con el eje y (x=0)
lm.coef_                    # coeficiente para cada caracteristica.
'''
        st.code(codigo)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        
        lm = LinearRegression()     # instancia del objeto LinearRegression
        lm.fit(X_train, y_train)    # entranemiento del modelo
        
        st.write('**intercept_**')
        codigo = 'st.text(lm.intercept_)'
        st.code(codigo)
        
        st.text(lm.intercept_)   # punto de intercepcion con el eje y (x=0)
        
        st.write('**coef_**')
        codigo = 'st.dataframe(pd.DataFrame(lm.coef_, index=X.columns), columns=[\'Coeficiente\'], width=400) # DataFrame con los coeficientes para cada caracteristica'     
        st.code(codigo)
        
        cdf = st.dataframe(pd.DataFrame(lm.coef_, index=X.columns, columns=['Coeficiente']), width=400)        # coeficiente para cada caracteristica   

        st.write('''Para un incremento de 1 metro cuadrado en Area Income, significa un aumento de U$S 21.57 en el precio e la casa.''')


        st.write('---')
        st.write('##### Predicciones del conjunto de test')
        st.write('''* **predict**: función que se utiliza para obtener predicciones de un modelo entrenado.      
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')

        codigo = '''predictions = lm.predict(X_test)'''
        st.code(codigo)
        
        predictions = lm.predict(X_test)
        st.write('Valores predichos de las casas: ')
        st.text(predictions)
        
        codigo = '''fig, ax = plt.subplots()
ax.scatter(x=y_test, y=predictions)
ax.set_xlabel('y_test')
ax.set_ylabel('Predicciones')
            
st.pyplot(fig)'''
        st.code(codigo)      
        
        
        with st.container(width=800):
            st.write('##### Scatter de y_test vs predicciones')
            fig, ax = plt.subplots()
            ax.scatter(x=y_test, y=predictions)
            ax.set_xlabel('y_test')
            ax.set_ylabel('Predicciones')
            
            st.pyplot(fig)
            
        st.write('---')
        st.write('##### Métricas de Evaluación')
        st.write('''**MAE**: Error absoluto medio - es una medida de la diferencia entre dos valores, permite saber que tan diferente es el vaor predicho y el valor real u observado.     
**MSE**: Error medio cuadratico - esta métrica es útil para saber que tan cerca es la línea de ajuste de la regresión a las observaciones. Entre más cercano a 0 es mejor.  
**RMSE**: Raíz del error medio cuadrado - es la raíz cuadrada del MSE.
''')    
        codigo = '''from sklearn import metrics
st.write(f'MAE: {metrics.mean_absolute_error(y_test, predictions)}')
st.write(f'MSE: {metrics.mean_squared_error(y_test, predictions)}')
st.write(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}') '''
        st.code(codigo)
                  
        st.write(f'MAE: {metrics.mean_absolute_error(y_test, predictions)}')
        st.write(f'MSE: {metrics.mean_squared_error(y_test, predictions)}')
        st.write(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')            
    
    
    
    # Ecommerce Customers
    if opcion_seleccionada == 'Ecommerce Customers':
        st.write('##### Data Frame con los datos de compras Ecommerce de clientes.')
        
        st.code('''# Carga de csv con los datos
df = pd.read_csv('DataFrames/Ecommerce Customers')
st.dataframe(df.head(10))       # head''')
        
        df = pd.read_csv('DataFrames/Ecommerce Customers')
        st.dataframe(df.head(10))       # head
        
        st.code('''# info
st.dataframe(df.info())''')     
        
        df.info(buf=buffer)             # info
        st.code(buffer.getvalue(), language='html')
    
        st.write('---')
        st.write('#### EDA')
        
        st.code('''# Seleccion de variables numericas
df_numericas = df.select_dtypes(include=['float64'])
st.dataframe(df_numericas.head())''')
        
        # Seleccion de variablees numericas
        df_numericas = df.select_dtypes(include=['float64'])
        st.dataframe(df_numericas.head())
    
        st.write('###### Visualización')
    
        st.code('''sns.set_style('whitegrid')
graf = sns.jointplot(data=df_numericas, x='Time on Website', y='Yearly Amount Spent')
st.pyplot(graf.figure)                         
''')  
 
        st.code('''sns.set_style('whitegrid')
graf = sns.jointplot(data=df_numericas, x='Time on App', y='Yearly Amount Spent')
st.pyplot(graf.figure)                         
''')   
  
        col_1, col_2 = st.columns(2)
            
        with col_1:
            with st.container(border=True):

                st.write('##### Time on Website vs Yearly Amount Spent')
                sns.set_style('whitegrid')
                graf = sns.jointplot(data=df_numericas, x='Time on Website', y='Yearly Amount Spent')
                st.pyplot(graf.figure) 

        with col_2:
            with st.container(border=True):
                st.write('##### Time on App vs Yearly Amount Spent')
                sns.set_style('whitegrid')
                graf = sns.jointplot(data=df_numericas, x='Time on App', y='Yearly Amount Spent')
                st.pyplot(graf.figure)         
            
        st.code('''sns.set_style('whitegrid')
graf = sns.jointplot(data=df_numericas, x='Time on App', y='Yearly Amount Spent',kind='hex')
st.pyplot(graf.figure)                   
''')    
        st.code('''sns.set_style('whitegrid')
graf = sns.lmplot(data=df_numericas, x='Yearly Amount Spent', y='Length of Membership')                 
st.pyplot(graf.figure) ''')    
            
        col_1, col_2 = st.columns(2)
        with col_1:
            with st.container(border=True):
                st.write('##### Time on app vs Lenght of Membership (2d hex)')
                sns.set_style('whitegrid')
                graf = sns.jointplot(data=df_numericas, x='Time on App', y='Length of Membership',kind='hex')
                st.pyplot(graf.figure)              
         
        with col_2:
             with st.container(border=True):
                st.write('##### Yearly Amount Spent vs Length of Membership')
                sns.set_style('whitegrid')
                graf = sns.lmplot(data=df_numericas, x='Yearly Amount Spent', y='Length of Membership') 
                st.pyplot(graf.figure) 
         
        st.write('---') 
        st.write('#### Separación de los datos del modelo') 

        st.write('''**train_test_split**: función que permite hacer una división de un conjunto de datos en dos bloques de entrenamiento (train) y prueba (test) de un modelo.  
Mediante el parámetro test_size, se pasa el % de los datos correspondientes a test.     
El parámetro random_state permite conseguir cierta repetición de los resultados.   
El parámetro shuffle realiza un reordenamiento aleatorio de los datos.             
''')
        
        st.code('''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression                
''')        
        
        st.code('''X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101, shuffle=True)
''')
        
        X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
        y = df['Yearly Amount Spent']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101, shuffle=True)

        st.write('---') 
        st.write('#### Entrenamiento del modelo')  
        st.write('''**fit**: función que permite entrenar un modelo para que aprenda a predecir etiquetas (y) a partir de características (X)''')
    
        st.code('''lm = LinearRegression()
lm.fit(X_train, y_train)   
lm.coef_             
''')
    
        lm = LinearRegression()
        lm.fit(X_train, y_train)

        cdf = st.dataframe(pd.DataFrame(lm.coef_, index=X.columns, columns=['Coeficiente']), width=400)        # coeficiente para cada caracteristica   
          
        st.write('El aumento de un año en la membresía del cliente, aumenta en U$S 61.28 lo gastado.')  
          
        st.write('---') 
        st.write('#### Predicciones del conjunto de test')
        st.write('''**predict**: función que se utiliza para obtener predicciones de un modelo entrenado.
Toma datos nuevos e invisibles como entrada y genera las predicciones del modelo para esos datos.''')            
        
        st.code('predictions = lm.predict(X_test) ')
          
        predictions = lm.predict(X_test)   
        
        st.code('''fig, ax = plt.subplots()
ax.scatter(x=y_test,y=predictions)  
ax.set_title('Scatter y_test vs predicciones')  
st.pyplot(fig)            
''')
        
        col_1, col_2 = st.columns(2)
        with col_1:
            with st.container(border=True):
        
                fig, ax = plt.subplots()
                ax.scatter(x=y_test,y=predictions)
                ax.set_title('Scatter y_test vs predicciones')
                st.pyplot(fig)
        

        st.write('---') 
        st.write('#### Evaluación del modelo')
        
        st.write('''**MAE**: Error absoluto medio - es una medida de la diferencia entre dos valores, permite saber que tan diferente es el valor predicho y el valor real u observado.     
**MSE**: Error medio cuadratico - esta métrica es útil para saber que tan cerca es la línea de ajuste de la regresión a las observaciones. Entre más cercano a 0 es mejor.  
**RMSE**: Raíz del error medio cuadrado - es la raíz cuadrada del MSE.
''')    
        codigo = '''from sklearn import metrics
        
st.write(f'MAE: {metrics.mean_absolute_error(y_test, predictions)}')
st.write(f'MSE: {metrics.mean_squared_error(y_test, predictions)}')
st.write(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}') '''
        st.code(codigo)
                  
        st.write(f'MAE: {metrics.mean_absolute_error(y_test, predictions)}')
        st.write(f'MSE: {metrics.mean_squared_error(y_test, predictions)}')
        st.write(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}') 
        
        
        st.code('''sns.set_style('whitegrid')
graf = sns.displot(data=y_test-predictions, bins=50, kde=True) 
st.pyplot(graf.figure)                 
''')
        
        col_1, col_2 = st.columns(2)
        with col_1:
            with st.container(border=True):
        
                sns.set_style('whitegrid')
                graf = sns.displot(data=y_test-predictions, bins=50, kde=True) 
                st.pyplot(graf.figure) 
         
        
        
        
        

        
            
def python():

    opciones_py = ['print','input','type','Conversión de Tipo','Operadores', 'Métodos de Cadenas (strings)','round',
                   'Módulo math','Módulo random','Módulo statistics','Listas','Tuplas','Sets','Diccionarios','range','Condicional If',
                   'Condicional in - not in', 'Ciclo for', 'Ciclo while','Funciones','lambda','zip','filter','map','reduce','Generadores','Excepciones'
                   ]
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_py)
        st.success(f'##### **{opcion_seleccionada}** ')
    
    # ------------------------------------------- PRINT() -------------------------------------------------------
    if opcion_seleccionada == 'print':
        st.write('**Definición**: Función que se utiliza para mostrar mensajes en pantalla.')
    
        
        imagen = Image.open('Imagenes/print_1.png')
        st.image(imagen, width=1466)
        
        imagen = Image.open('Imagenes/print_2.png')
        st.image(imagen, width=1463)     
        
        imagen = Image.open('Imagenes/print_3.png')
        st.image(imagen, width=1465)         
        
        st.write('---')
        st.write('''
            **format()** es una funcion que permite formatear una cadena de texto.  
            **f-string** permite concatenar diferentes tipos de datos dentro de un string
        ''') 
        imagen = Image.open('Imagenes/format.png')
        st.image(imagen, width=1470)           
        
        st.write('---')
        st.write('##### Referencias externas')
        url = 'https://www.w3schools.com/python/ref_func_print.asp'
        st.page_link(url, label='W3 Schools: Funcion print() de Python')
        url = 'https://www.w3schools.com/python/ref_string_format.asp'
        st.page_link(url, label='W3 Schools: Funcion String format() de Python')  
      

    if opcion_seleccionada == 'input':   
        st.write('Funcion que permite al programa solicitar y recibir datos del usuario a través de la consola')
        imagen = Image.open('Imagenes/input.png')
        st.image(imagen, width=1469)   
        
        st.write('---')
        st.write('##### Referencias externas')
        url = 'https://www.w3schools.com/python/ref_string_format.asp'
        st.page_link(url, label='W3 Schools: Funcion String format() de Python')              
        
        
    if opcion_seleccionada == 'type':      
        st.write('Funcion que devuelve el tipo de dato')
        imagen = Image.open('Imagenes/type.png')
        st.image(imagen, width=1467)
 
        st.write('---')
        st.write('##### Referencias externas')
        url = 'https://www.w3schools.com/python/ref_func_type.asp'
        st.page_link(url, label='W3 Schools: Funcion type() en Python')     
 

    if opcion_seleccionada == 'Conversión de Tipo':   
        st.write('Funciones que permiten convertir el tipo de dato')
        st.write('''
            **str()** convierte un numero a un string.     
            **float()** convierte un entero o string a un float.        
            **int()** convierte un float o string a un int.
        ''')    
        
        imagen = Image.open('Imagenes/conv_tipo.png')
        st.image(imagen, width=1465)
        
        st.write('---')
        st.write('##### Referencias externas')
        url = 'https://www.w3schools.com/python/ref_func_str.asp'
        st.page_link(url, label='W3 Schools: Funcion str() en Python')     
        url = 'https://www.w3schools.com/python/ref_func_int.asp'
        st.page_link(url, label='W3 Schools: Funcion int() en Python')  
        url = 'https://www.w3schools.com/python/ref_func_float.asp'
        st.page_link(url, label='W3 Schools: Funcion float() en Python')         
        
    if opcion_seleccionada == 'Operadores':      
        st.write('**Operadores Aritmeticos:** Suma, Resta, Multiplicacion, Division, Modulo, Divison entera, Exponente, Raiz cuadrada, Suma de complejos.')
        imagen = Image.open('Imagenes/operadores_1.png')
        st.image(imagen, width=1465)
 
        imagen = Image.open('Imagenes/operadores_2.png')
        st.image(imagen, width=1465)
 
        st.write('---')
        st.write('##### Referencias externas')
        url = 'https://ellibrodepython.com/operadores-aritmeticos'
        st.page_link(url, label='El Libreo de Python: Operadores Aritmeticos')           
        
    if opcion_seleccionada == 'Métodos de Cadenas (strings)':      
        st.write('**Métodos:** upper, lower, capitalize, title, swapcase, len, replace, lstrip, rstrip, strip, find, isdigit, isalum, isalpha')
        imagen = Image.open('Imagenes/cadena_1.png')
        st.image(imagen, width=1472)
        
        imagen = Image.open('Imagenes/cadena_2.png')
        st.image(imagen, width=1469)
        
        imagen = Image.open('Imagenes/cadena_3.png')
        st.image(imagen, width=1468)

        st.divider()
        st.write('**Indexación de Strings**')
        st.write('''
        * [0:3] : devuelve a partir de la posición 0 hasta la 2 (la posición 3 no se incluye)     
        * [::2] : devuelve desde la primera a la ultima posición con un paso de 2      
        ''')
        imagen = Image.open('Imagenes/cadena_4.png')
        st.image(imagen, width=1470)
        st.write('---')
        
        st.write('**split()** funcion que permite dividir una cadena en subcadenas, devuelve una lista')
        imagen = Image.open('Imagenes/cadena_5.png')
        st.image(imagen, width=1469)
        
        st.write('**join()** funcion que permite unir subcadenas para formar una cadena')
        imagen = Image.open('Imagenes/cadena_6.png')
        st.image(imagen, width=1468)
 
        st.write('---')
        st.write('##### Referencias externas')  
       
    if opcion_seleccionada == 'round':      
        st.write('Funcion que permite redondear un float')
        imagen = Image.open('Imagenes/round.png')
        st.image(imagen, width=1471)
 
        st.write('---')
        st.write('##### Referencias externas')
  
    if opcion_seleccionada == 'Módulo math':      
        st.write('**Métodos:** ceil, floor, sqrt, pow')
        imagen = Image.open('Imagenes/math.png')
        st.image(imagen, width=1471)
 
        st.write('---')
        st.write('##### Referencias externas')  
     
    if opcion_seleccionada == 'Módulo random':      
        st.write('**Métodos:** random, randint, randrange')
        imagen = Image.open('Imagenes/random_1.png')
        st.image(imagen, width=1471)
 
        st.divider()
        st.write('**shuffle()** reordena los items de una lista de forma aleatoria') 
        imagen = Image.open('Imagenes/random_2.png')
        st.image(imagen, width=1469)
 
        st.divider()
        st.write('**choice()** retorna un elemento de una lista') 
        imagen = Image.open('Imagenes/random_3.png')
        st.image(imagen, width=1469)
 
 
        st.divider()
        st.write('##### Referencias externas')       
     
    if opcion_seleccionada == 'Módulo statistics':      
        st.write('**Métodos:** mean, median, mode, stdev, pstdev, variance, pvariance')
        imagen = Image.open('Imagenes/statistics.png')
        st.image(imagen, width=1467)
 
        st.divider()
        st.write('##### Referencias externas')       
     
 
    if opcion_seleccionada == 'Listas':      
        imagen = Image.open('Imagenes/lista_1.png')
        st.image(imagen, width=1468)
        imagen = Image.open('Imagenes/lista_2.png')
        st.image(imagen, width=1468) 
        imagen = Image.open('Imagenes/lista_3.png')
        st.image(imagen, width=1470) 
        imagen = Image.open('Imagenes/lista_4.png')
        st.image(imagen, width=1479)  
        imagen = Image.open('Imagenes/lista_5.png')
        st.image(imagen, width=1478)  
        imagen = Image.open('Imagenes/lista_6.png')
        st.image(imagen, width=1480)  

        st.write('---')
        st.write('**Listas de Comprension**')
        imagen = Image.open('Imagenes/lista_7.png')
        st.image(imagen, width=1480)

        st.write('---')
        st.write('##### Referencias externas')     
  
    if opcion_seleccionada == 'Tuplas':    
        imagen = Image.open('Imagenes/tupla.png')
        st.image(imagen, width=1477)
        
    if opcion_seleccionada == 'Sets':    
        imagen = Image.open('Imagenes/set_1.png')
        st.image(imagen, width=1475)
    
        st.write('---')
        st.write('**union()** se utiliza para combinar los elementos de dos conjuntos sin duplicar ningun elemento') 
        imagen = Image.open('Imagenes/set_2.png')
        st.image(imagen, width=1479)    
    
        st.write('---')
        st.write('**intersection()** se utiliza para encontrar elementos entre dos conjuntos') 
        imagen = Image.open('Imagenes/set_3.png')
        st.image(imagen, width=1479)       
    
    if opcion_seleccionada == 'Diccionarios':     
        imagen = Image.open('Imagenes/diccionario_1.png')
        st.image(imagen, width=1478)
        imagen = Image.open('Imagenes/diccionario_2.png')
        st.image(imagen, width=1480)
        imagen = Image.open('Imagenes/diccionario_3.png')
        st.image(imagen, width=1480)       
               
        st.write('---')
        st.write('**Coprension de Diccionarios**')
        imagen = Image.open('Imagenes/diccionario_4.png')
        st.image(imagen, width=1479)
            
    if opcion_seleccionada == 'zip':     
        st.write('Metodo que toma iterables y los combina en una secuencia de tuplas')

        imagen = Image.open('Imagenes/zip.png')
        st.image(imagen, width=1479)
 
        st.write('---')
        st.write('##### Referencias externas')     
 
    if opcion_seleccionada == 'Condicional If':     
        imagen = Image.open('Imagenes/if_1.png')
        st.image(imagen, width=1479)  
  
        imagen = Image.open('Imagenes/if_2.png')
        st.image(imagen, width=1479)    
   
    if opcion_seleccionada == 'Condicional in - not in':     
        imagen = Image.open('Imagenes/in.png')
        st.image(imagen, width=1479)      
   
    if opcion_seleccionada == 'Ciclo for':     
        imagen = Image.open('Imagenes/for.png')
        st.image(imagen, width=1480)      
    
    if opcion_seleccionada == 'Ciclo while':     
        imagen = Image.open('Imagenes/while.png')
        st.image(imagen, width=1479)         
       
    if opcion_seleccionada == 'range':     
        st.write('Función que retorna una secuencia de numeros')

        imagen = Image.open('Imagenes/range.png')
        st.image(imagen, width=1479)
 
        st.write('---')
        st.write('##### Referencias externas')              
       
    if opcion_seleccionada == 'Funciones':     
        imagen = Image.open('Imagenes/funciones_1.png')
        st.image(imagen, width=1477)      
        imagen = Image.open('Imagenes/funciones_2.png')
        st.image(imagen, width=1480)            
        imagen = Image.open('Imagenes/funciones_3.png')
        st.image(imagen, width=1478)            
        imagen = Image.open('Imagenes/funciones_4.png')
        st.image(imagen, width=1481)            
        
        st.write('---')
        st.write('**Funciones de numeros()**') 
        imagen = Image.open('Imagenes/funciones_5.png')
        st.image(imagen, width=1480)          
        
    if opcion_seleccionada == 'lambda':         
        st.write('Utilizado para crear funciones anonimas')

        imagen = Image.open('Imagenes/lambda.png')
        st.image(imagen, width=1478)
 
        st.write('---')
        st.write('##### Referencias externas')          
        
    if opcion_seleccionada == 'filter':         
        st.write('''Verifica que los elementos de una secuencia cumplan una condicion, 
        devolviendo un iterador con los elementos que cumplen dicha condicion.
    ''')

        imagen = Image.open('Imagenes/filter.png')
        st.image(imagen, width=1479)
 
        st.write('---')
        st.write('##### Referencias externas')    
             
    if opcion_seleccionada == 'map':         
        st.write('''Aplica una funcion a cada elemento de un iterable devolviendo una lista con los resultados.
    ''')

        imagen = Image.open('Imagenes/map.png')
        st.image(imagen, width=1478)
 
        st.write('---')
        st.write('##### Referencias externas')            
 
    if opcion_seleccionada == 'reduce':         
        st.write('''Aplica una funcion que calcula el produco de todos los elementos de una lista.
    ''')

        imagen = Image.open('Imagenes/reduce.png')
        st.image(imagen, width=1480)
 
        st.write('---')
        st.write('##### Referencias externas')    
 
    if opcion_seleccionada == 'Generadores':         
        st.write('''Es una funcion que devuelve varios valores en tiempo de ejecucion.
    ''')

        imagen = Image.open('Imagenes/generador.png')
        st.image(imagen, width=1480)
 
        st.write('---')
        st.write('##### Referencias externas')  
   
    if opcion_seleccionada == 'Excepciones':         

        imagen = Image.open('Imagenes/excepcion.png')
        st.image(imagen, width=1481)
 
        st.write('---')
        st.write('##### Referencias externas')     
        
        
def numpy():
    opciones_np = ['array','random','Operaciones con Arrays','Valores Estadisticos','Transponer un vector y una matriz',
                   'Operaciones Algebraicas','Filtrado de Datos','Valores Faltantes','Importación y exportación de datos']
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_np)
        st.success(f'##### **{opcion_seleccionada}** ')

    if opcion_seleccionada == 'array':         
        st.write('''Es una funcion que crear un array (vector o matriz).
    ''')

        imagen = Image.open('Imagenes/numpy_1.png')
        st.image(imagen, width=1479)
        imagen = Image.open('Imagenes/numpy_2.png')
        st.image(imagen, width=1480)
        imagen = Image.open('Imagenes/numpy_3.png')
        st.image(imagen, width=1477)
        imagen = Image.open('Imagenes/numpy_4.png')
        st.image(imagen, width=1478)
 
        st.write('---')
        st.write('**arange()** crea un vector con una cantidad de numeros definidos.') 
        imagen = Image.open('Imagenes/numpy_5.png')
        st.image(imagen, width=1478) 
 
        st.write('---')
        st.write('**linspace()** crea un vector con valores separados por una distancia definida') 
        imagen = Image.open('Imagenes/numpy_6.png')
        st.image(imagen, width=1477)  

        st.write('---')
        st.write('**reshape()** permite redimensionar un array') 
        imagen = Image.open('Imagenes/numpy_8.png')
        st.image(imagen, width=1475)  
 
 
        st.write('---')
        st.write('##### Referencias externas')    

    if opcion_seleccionada == 'random':         
        st.write('''Funcion que crea numeros pseudoaleatorios.
    ''')
        
        imagen = Image.open('Imagenes/numpy_7.png')
        st.image(imagen, width=1478)

        st.write('---')
        st.write('##### Referencias externas')  
        
    if opcion_seleccionada == 'Operaciones con Arrays':         
        imagen = Image.open('Imagenes/numpy_9.png')
        st.image(imagen, width=1480)
        imagen = Image.open('Imagenes/numpy_10.png')
        st.image(imagen, width=1479)        
        imagen = Image.open('Imagenes/numpy_11.png')
        st.image(imagen, width=1478)
        imagen = Image.open('Imagenes/numpy_12.png')
        st.image(imagen, width=1476)
       
    if opcion_seleccionada == 'Valores Estadisticos':         
        imagen = Image.open('Imagenes/numpy_13.png')
        st.image(imagen, width=1478)      
       
    if opcion_seleccionada == 'Transponer un vector y una matriz':         
        imagen = Image.open('Imagenes/numpy_14.png')
        st.image(imagen, width=1478)       
        
    if opcion_seleccionada == 'Operaciones Algebraicas':         
        imagen = Image.open('Imagenes/numpy_15.png')
        st.image(imagen, width=1480)
        imagen = Image.open('Imagenes/numpy_16.png')
        st.image(imagen, width=1479)

    if opcion_seleccionada == 'Filtrado de Datos':         
        imagen = Image.open('Imagenes/numpy_17.png')
        st.image(imagen, width=1477)
        


    if opcion_seleccionada == 'Valores Faltantes':         
        imagen = Image.open('Imagenes/numpy_18.png')
        st.image(imagen, width=1482)

    if opcion_seleccionada == 'Importación y exportación de datos':         
        imagen = Image.open('Imagenes/numpy_19.png')
        st.image(imagen, width=1479)


        st.write('---')
        st.write('##### Referencias externas')        
        
        
def pandas():
    
    
    
        buffer = io.StringIO()
    
        opciones_pd = ['Series','Data Frames','Conversión de Tipos','Fusionar, Combinar y Concatenar Data Frames',
                    'Respaldos']
    
    
        df_ventas = pd.read_csv('DataFrames/Resumen_ventas.csv', index_col=0)
    
    
        col1, col2 = st.columns([2,2])
    
        with col1:
                opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_pd)
                st.success(f'##### **{opcion_seleccionada}** ')

        if opcion_seleccionada == 'Series':         
                st.write('''Una Serie es una estructura de datos unidimensional que puede contener cualquier tipo de datos.
Es como una columna de una tabla.''')
        
                st.write('##### Creación de una Serie')
                imagen = Image.open('Imagenes/pandas_1.png')
                st.image(imagen, width=1478)
                imagen = Image.open('Imagenes/pandas_2.png')
                st.image(imagen, width=1478)        
                st.write('---')
                st.write('##### Acceso a los valores de una Serie')
                imagen = Image.open('Imagenes/pandas_3.png')
                st.image(imagen, width=1479) 
                st.write('---')
                st.write('##### Operaciones con Series')
                imagen = Image.open('Imagenes/pandas_4.png')
                st.image(imagen, width=1479) 
                imagen = Image.open('Imagenes/pandas_5.png')
                st.image(imagen, width=1478) 
                imagen = Image.open('Imagenes/pandas_6.png')
                st.image(imagen, width=1477) 
                st.write('---')
                st.write('##### Filtrado')
                imagen = Image.open('Imagenes/pandas_7.png')
                st.image(imagen, width=1479) 
                st.write('##### Valores faltantes')
                imagen = Image.open('Imagenes/pandas_8.png')
                st.image(imagen, width=1479) 


        if opcion_seleccionada == 'Data Frames':         
                st.write('''Un DataFrame es una estructura de datos bidimensional con etiquetas que se asemeja a una hoja de cálculo o una tabla
de base de datos.   
Se compone de filas y columnas, donde cada columna puede contener un tipo de dato diferente.''')
                st.write('##### Creación de un DataFrame')
        
                # Creacion de un DataFrame (utilizando un diccionario)
                st.code('''# Creacion de un DataFrame (utilizando un diccionario)
datos = {
        'Nombre':['Ana','Luis','Carlos','Sara'], 
        'Edad':[25,30,22,27], 
        'Ciudad':['Madrid','Barcelona','Valencia','Bilbao']
}

indice = [1,2,3,4]

df = pd.DataFrame(data=datos, index=indice)
df.head()''')        
        
                datos = {
                        'Nombre':['Ana','Luis','Carlos','Sara'], 
                        'Edad':[25,30,22,27], 
                        'Ciudad':['Madrid','Barcelona','Valencia','Bilbao']
                }

                indice = [1,2,3,4]
        
                df = pd.DataFrame(data=datos, index=indice)
                st.dataframe(df.head())        
        
        
                st.write('''**set_index**: se utiliza para convertir una o mas columnas existentes en el indice de un DataFrame.''')
        
                st.code('''# set_index 
df.set_index('Nombre', inplace=True)
df.head()''')
       
                df.set_index('Nombre', inplace=True)
                st.dataframe(df.head())
                
                st.divider()

                st.write('##### DataSet')
                st.write('''Un DataSet son los datos que estan organizados de cierta manera en un archivo txt, csv, xlsx, etc.''')
        
                st.write('''**Parámetros del read_csv**
                  
* sep -> el caracter utilizado para separar los valores (delimitador). El predeterminado es la coma ','.        
* header -> la fila que se usará como encabezado, header=0 (primera fila) o header=None.
* names -> una lista de nombres de columna para usar en caso de que el archivo no tenga encabezado.
* index_col -> especifica la columna a usar como índice del DataFrame.
* na_values -> se utiliza para especificar que valores deben interpretarse como valores faltantes (NaN) al cargarlo en un DataFrame. 
Se pueden pasar una lista de cadenas (n/a, ---, ?, etc.) ademas de los valores predeterminados como '', 'NULL', 'NA', etc.''')
        
                st.divider()
                st.write('##### Cargar datos desde un archivos csv')
                st.code('''df_tips = pd.read_csv('DataFrames/tips.csv')
df_tips''')
        
                df_tips = pd.read_csv('DataFrames/tips.csv')
                st.dataframe(df_tips.head(10))
        
                st.divider()
                st.write('##### Renombrar columnas')
                df_tips.columns = ['Total Factura','Propina','Sexo','Fumador','Dia','Horario','Nro Clientes']
        
                st.code('df_tips.columns = [\'Total Factura\',\'Propina\',\'Sexo\',\'Fumador\',\'Dia\',\'Horario\',\'Nro Clientes\']')
                st.dataframe(df_tips.head())
        
                st.write('**Renombrar columnas especificas**')
                
                df_tips = pd.read_csv('DataFrames/tips.csv')
        
                st.code('''datos = {'total_bill':'Total Facturas', 'tip':'Propina'}
df_tips.rename(columns=datos, inplace=True)''')
        
                datos = {'total_bill':'Total Facturas', 'tip':'Propina'}
                df_tips.rename(columns=datos, inplace=True)
                st.dataframe(df_tips.head())
        
        
                st.divider()
                st.write('##### Cargar datos desde un archivos xls')

                st.code('''df_temp = pd.read_excel('DataFrames/Temperaturas.xlsx', sheet_name='Sheet1')
df_temp''')
        
                df_temp = pd.read_excel('DataFrames/Temperaturas.xlsx', sheet_name='Sheet1')
                st.dataframe(df_temp.head(10))
                
                
                st.divider()
                st.write('##### Cargar un archivo txt dese una url')

                st.code('''url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
df_diabetes = pd.readcsv(url, sep='\ t')''')
                
                url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
                df_diabetes = pd.read_csv(url, sep='\t')
                st.dataframe(df_diabetes.head(10))        
        
        
                st.divider()

                st.write('##### Acceder a columnas y filas') 

                st.code('''df_ventas = pd.read_excel('DataFrames/Ventas.csv', sep=';')
df_ventas''')
                
                df_ventas = pd.read_csv('DataFrames/Ventas.csv', sep=';')
                st.dataframe(df_ventas.head(20))
                        


                st.write('**Acceder a una columna**')
                
                
                st.code('''Venta_Motos = df_ventas[['Total Venta Motorcycles']]
Venta_Motos.head()''')
                
                Venta_Motos = df_ventas[['Total Venta Motorcycles']]
                st.dataframe(Venta_Motos.head())
                
                st.write('''**Acceder por el metodo loc**      
Se utiliza para seleccionar datos de un DataFrame utilizando etiquetas (nombres) de fila y columna''')
                
                
                st.code('''# Acceder a una fila especifica
fila_3 = df_ventas.loc[[3]]
fila_3''')
                
                fila_3 = df_ventas.loc[[3]]
                fila_3
                
                st.code('''# Acceder a  mas de una fila (pasando una lista)
filas = df_ventas.loc[[2,5,8]]
filas''')        
                filas = df_ventas.loc[[2,5,8]]
                filas
     
           
                st.code('''# Acceder a  filas segun una condicion
filtro = df_ventas['Total Venta Classic Cars'] > 200000
autos = df_ventas.loc[filtro]
autos''')                   
                
                filtro = df_ventas['Total Venta Classic Cars'] > 200000
                autos = df_ventas.loc[filtro]
                autos 
                
                st.code('''# Acceder a un rango de filas y columnas especificas
columnas = df_ventas.loc[2:5, ['Total Venta Planes','Total Venta Ships']]
columnas''')                   
                        
                columnas = df_ventas.loc[2:5, ['Total Venta Planes','Total Venta Ships']]
                columnas           
        
                st.divider()
           
                st.write('''**Acceder por el metodo iloc**      
Permite seleccionar los datos en un DataFrame utilizando posiciones enteras ''')

        
                st.code('''# # Acceder a las 3 primeras filas y a las columnas con indice 2 y 6
datos = df_ventas.iloc[:3, [3,5]]
datos''')   
        
        
                datos = df_ventas.iloc[:3, [3,5]]
                datos
        
                st.divider()
                
                
                st.write('##### Transponer un DataFrame')      
        
                st.code('''df_ventas_t = df_ventas.T
df_ventas_t''')   
  
                df_ventas_t = df_ventas.T
                st.dataframe(df_ventas_t)
                
                
                st.divider()


                st.write('**value_counts()**: obtiene la cantidad de una variable agrupada por categoría')              
        
        
                df_tips = pd.read_csv('DataFrames/tips.csv')
                st.dataframe(df_tips.head(10))
        
       
                st.code('''df_tips[\'day\'].value_counts()
dias''')       
        
                dias = df_tips['day'].value_counts()
                dias

                st.code('''df_tips[\'sex\'].value_counts()
sex''')   
                sexo = df_tips['sex'].value_counts()
                sexo

                st.code('''df_tips[\'time\'].value_counts()
time''')   
                time = df_tips['time'].value_counts()
                time


                st.divider()
        
                df_tips.columns = ['Total Factura','Propina','Sexo','Fumador','Dia','Horario','Nro Clientes']
                
                st.write('##### Reemplazar valores') 
                st.write('''**replace()**: permite reemplazar los valores por otra etiqueta o crear una nueva columna en base a los valores de la otra columna.
No altera valores no especificados en el diccionario.''')

                st.code('''dia = {'Sat':'Sabado', 'Sun':'Domingo', 'Thur':'Jueves', 'Fri':'Viernes'
df_tips['Dia'].replace(dia, inplace=True)}''')


                dia = {'Sat':'Sabado', 'Sun':'Domingo', 'Thur':'Jueves', 'Fri':'Viernes'}
                df_tips['Dia'].replace(dia, inplace=True)

                st.code('''sexo = {'Female':'Mujer', 'Male':'Hombre'}
df_tips['Sexo''].replace(sexo, inplace=True)}''')

                sexo = {'Female':'Mujer', 'Male':'Hombre'}
                df_tips['Sexo'].replace(sexo, inplace=True)
                
                st.code('''horario = {'Dinner':'Cena', 'Lunch':'Almuerzo'}
df_tips['Horario''].replace(horario, inplace=True)}''')        
                
                horario = {'Dinner':'Cena', 'Lunch':'Almuerzo'}
                df_tips['Horario'].replace(horario, inplace=True)
        
                st.dataframe(df_tips.head(10))        
                
                st.divider()        
                st.write('''**map()**: permite substituir cada valor de una columna por otro valor basandose en un diccionario, una funcion u otra columna.
Si un valor no esta en el diccionario lo convierte en NaN''')

                st.code('''datos = {3:'Tres', 4:'Cuatro'}
df_tips['Clientes_G'] = df_tips['Nro Clientes'].map(datos)
df_tips''')

                datos  = {3:'Tres', 4:'Cuatro'}

                df_tips['Clientes_G'] = df_tips['Nro Clientes'].map(datos)
                st.dataframe(df_tips.head(20))
        


                st.divider()

        
                st.write('''**apply()**: Es una herramienta de propósito general para aplicar una función a lo largo de un eje (filas o columnas) de un DataFrame o a cada elemento de una Serie.
Se utiliza para realizar transformaciones, cálculos y lógica condicional compleja, siendo una alternativa más eficiente y limpia que los bucles.
Para usarlo, se le pasa la función a aplicar, y el parámetro axis determina si se aplica a las columnas (0) o a las filas (1)''')
                
                st.code('''def categorizar(fila):
        if fila['Nro Clientes'] > 2:
                return 'Familia'
        else:
                return 'Pareja'
                
df_tips['Tipo Clientes'] = df_tips.apply(categorizar, axis=1)
df_tips''')
                
                
                def categorizar(fila):
                        if fila['Nro Clientes'] > 2:
                                return 'Familia'
                        else:
                                return 'Pareja'

                df_tips['Tipo Clientes'] = df_tips.apply(categorizar, axis=1)
                
                st.dataframe(df_tips.head(10))
        
       
        
                st.divider()
                st.write('##### Filtrado de datos')     

                st.code('''filtro = df_tips['Dia'] == 'Sabado'
df_tips_sabado = df_tips[filtro]               
df_tips_sabado.head()''')

                filtro = df_tips['Dia'] == 'Sabado'
                df_tips_sabado = df_tips[filtro]
                st.dataframe(df_tips_sabado.head(10))


                st.write('**Multiples coniciones: & (and), | (or), ~ (not)**')  
                
                st.code('''filtro = (df_tips['Dia'] == 'Jueves') | (df_tips['Dia'] == 'Viernes')
df_tips_jv = df_tips[filtro]                
df_tips_jv.head()''')
        
                filtro = (df_tips['Dia'] == 'Jueves') | (df_tips['Dia'] == 'Viernes')
                df_tips_jv = df_tips[filtro]
                st.dataframe(df_tips_jv.head(10))
        
        
                # Otra forma
                st.write('Otra forma de filtrado')
        
                st.code('''filtro = df_tips['Dia'].isin(['Jueves','Viernes'])
df_tips_jv = df_tips[filtro]
df_tips_jv.head()''')
                
                filtro = df_tips['Dia'].isin(['Jueves','Viernes'])
                df_tips_jv = df_tips[filtro]
                st.dataframe(df_tips_jv.head(10))
                

                st.divider()
                st.write('##### Reemplazar valores nulos.')  
        
                df_ventas = pd.read_csv('DataFrames/Ventas.csv', sep=';')
                st.dataframe(df_ventas.head(10))     

                
                st.write('''**fillna()**: remplaza los valores NaN por otro valor.''') 
                
                st.code('''df_ventas['Total Venta Planes'] = df_ventas['Total Venta Planes'].fillna(0)''')
                
                df_ventas['Total Venta Planes'] = df_ventas['Total Venta Planes'].fillna(0)
                
        
        
                st.code('''mediana = df_ventas['Total Venta Trains'].median()
df_ventas['Total Venta Trains'] = df_ventas['Total Venta Trains'].fillna(mediana)''')
                
                mediana = df_ventas['Total Venta Trains'].median()
                df_ventas['Total Venta Trains'] = df_ventas['Total Venta Trains'].fillna(mediana)
                
                st.dataframe(df_ventas.head(10))

                st.divider()
                st.write('##### dropna(): Eliminar valores faltantes.') 
                
                st.code('''# Se eliminan las filas que tengan algun valor NaN
df_ventas.dropna()''')
                df_ventas.dropna()

                st.code('''# Se eliminan las filas que tengan minimo 2 valores NaN
df_ventas.dropna(thresh=2)''')        
                df_ventas.dropna(thresh=2)
                
                st.code('''# Las columnas que contengan al menos algun valor NaN
df_ventas.dropna(axis=1)''')                
                df_ventas.dropna(axis=1)
        
        


                st.divider()
                st.write('##### Eliminar columna')  
                
                st.write('**drop()**: elimina una columna del DataFrame, para eliminar mas de una columna se pasa una lista como parametro.')
                
                st.code('''df_ventas.drop('Total Venta Vintage Cars', axis=1, inplace=True)
df_ventas.head()''')
                
                df_ventas.drop('Total Venta Vintage Cars', axis=1, inplace=True)
                st.dataframe(df_ventas.head(10))
                
                st.divider()
                st.write('##### Eliminar fila')
                
                
                st.code('''df_ventas.drop(3, axis=0, inplace=True)
df_ventas.head()''')        
                df_ventas.drop(3, axis=0, inplace=True)                
                st.dataframe(df_ventas.head(10))
                
                st.divider()
                st.write('##### Eliminar filas duplicadas')  
                
                st.code('''df_ventas.drop_duplicates(subset='Total Venta Classic Cars')''')
                df_ventas.drop_duplicates(subset='Total Venta Classic Cars')
        
        
                st.divider()
                
                

                
                
                st.write('##### Agrupación')  
                
                df_tips = pd.read_csv('DataFrames/tips.csv')
                st.dataframe(df_tips.head(10))
                
                st.code('''# Agrupacion por dia mostrando la cantidad.
group_day = df_tips.groupby('day')['total_bill'].count()                
group_day''')
                
                group_day = df_tips.groupby('day')['total_bill'].count()
                group_day
                
                
                st.code('''# Agrupacion por dia mostrando la cantidad utilizando la funcion size (toma en cuenta los valores NaN).
group_day_size = df_tips.groupby('day').size()                
group_day_size''')                
                group_day_size = df_tips.groupby('day').size()
                group_day_size
                
                
                st.code('''# Agrupacion por dia mostrando la media de total_bill.
group_day = df_tips.groupby('day')['total_bill'].mean()                
group_day''')
                
                group_day = df_tips.groupby('day')['total_bill'].mean()
                group_day     
        
        
                st.code('''# Agrupacion por dia mostrando media, mediana, desviacion standard
group_day = df_tips.groupby('day')['total_bill'].agg(['mean','median','std'])              
group_day''')
        
                group_day = df_tips.groupby('day')['total_bill'].agg(['mean','median','std'])
                group_day           

                
                st.divider()
                
                st.write('##### Ordenar') 
                
                st.code('''# sort_values(): ordena una DataFrame por una columna de menor a mayor.
df_tips_orden = df_tips.sort_values(by='total_bill')               
df_tips_orden.head()''')
                
                df_tips_orden = df_tips.sort_values(by='total_bill')
                st.dataframe(df_tips_orden.head(10))
                
                st.code('''# Orden descendente
df_tips_orden = df_tips.sort_values(by='total_bill', ascending=False)               
df_tips_orden.head()''')        
                
                df_tips_orden = df_tips.sort_values(by='total_bill', ascending=False)
                st.dataframe(df_tips_orden.head(10))        
        
    
                st.code('''# Orden por mas de una columna
df_tips_orden = df_tips.sort_values(by=['size','total_bill'], ascending=[False, False])              
df_tips_orden.head()''')           
                
                df_tips_orden = df_tips.sort_values(by=['size','total_bill'], ascending=[False, False])
                st.dataframe(df_tips_orden.head(10))      
        
        
 
        
        if opcion_seleccionada == 'Conversión de Tipos':  
                
                st.code('''df_tips = pd.read_csv('DataFrames/tips.csv')
df_tips.head()''')
        
                df_tips = pd.read_csv('DataFrames/tips.csv')
                st.dataframe(df_tips.head(10))            
            
            
                st.divider()
                st.write('##### Devolver Tipo de Dato')   
                st.write('**dtype()**: devuelve el tipo de dato.')
                        
                st.code('''df_tips['sex'].dtype''')
                df_tips['sex'].dtype
                
                st.code('''df_tips['time'].dtype''')
                df_tips['time'].dtype                
                
                st.code('''df_tips['total_bill'].dtype''')
                df_tips['total_bill'].dtype               
            
                st.write('**dtypes()**: devuelve el tipo de dato de todos los campos.')
                st.code('''data_types = df_tips.dtypes
data_types''')
                data_types = df_tips.dtypes
                data_types
             
             
                st.write('##### Seleccionar columnas con el tipo de dato indicado')   
                st.write('**select_dtypes()**: seleccion por tipo de datos.')             
                
                st.code('''df_tips_numericos = df_tips.select_dtypes(include=['float64','int64'])
df_tips_numericos.head()''')        

                df_tips_numericos = df_tips.select_dtypes(include=['float64','int64'])
                st.dataframe(df_tips_numericos.head(10))
             
             
                st.divider()
                
                st.write('##### Convertir a tipo numerico')  
                st.write('''**to_numeric()**: convierte un valor de tipo object a numerico (por defecto float).         
errores='coerce': si encuentra un valor que no puede convertir a numero, lo reemplaza con un NaN.''') 
                
                
                st.code('''df_tips['columna'] = pd.to_numeric(df_tips['columna'], errors='coerce').astype('Int64')''')
                  
                
                st.write('''**astype()**: convierte un valor de tipo object a entero (int).''')                
                st.code('''df_tips['columna'] = df_tips['columna'].astype(int)''')
                
                   
                st.divider()

                st.write('##### Convertir variables categoricas a binarias')
                st.write('''**getdummies()**: se utiliza para convertir variables categóricas en variables ficticias o binarias (con valores de 0 o 1).   
                Este proceso, también conocido como codificación one-hot, es fundamental para preparar datos para algoritmos de aprendizaje automático que requieren entradas numéricas.    
                La función crea nuevas columnas para cada categoría única en la variable original, indicando la presencia (1) o ausencia (0) de esa categoría en cada fila''')

                st.write('**drop_first**=True: elimina la primera categoría para evitar la multicolinealidad')
                
                
                df_tips = pd.read_csv('DataFrames/tips.csv')
                
                st.code('''df_tips['sex'] = pd.get_dummies(df_tips['sex'], dtype=int, drop_first=True)
df_tips['smoker'] = pd.get_dummies(df_tips['smoker'], dtype=int, drop_first=True)                        
df_tips['time'] = pd.get_dummies(df_tips['time'], dtype=int, drop_first=True)''')
                
                df_tips['sex'] = pd.get_dummies(df_tips['sex'], dtype=int, drop_first=True)
                df_tips['smoker'] = pd.get_dummies(df_tips['smoker'], dtype=int, drop_first=True)
                df_tips['time'] = pd.get_dummies(df_tips['time'], dtype=int, drop_first=True)
                
                st.dataframe(df_tips.head(10))
                
                st.divider()
                st.write('##### Convertir a tipo fecha') 
                
                st.write('**to_datetime()**: convierte valores de una columna tipo object a un tipo de dato DateTime')
                st.code('''df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')''')
                
                
                st.divider()   

                st.write('##### Obtener años/mes/dia de un campo DateTime')  
                
                st.code('''df['Anio'] = df['Fecha'].dt.year
df['Mes'] = df['Mes'].dt.month                        
df['Dia'] = df['Dia'].dt.day''')

                 
                st.write('**Dia de la semana en numero**')
                st.code('''df['dia_semana'] = df['Fecha'].dt.day_of_week''')
                
  

        if opcion_seleccionada == 'Fusionar, Combinar y Concatenar Data Frames': 
                st.write('''
                **merge():** fusiona dos DataFarmes basandose en valores comunes de una o mas columnas.  
                ''') 
                
                st.code('''datos_1 = {'Id':[1,2,3], 'Nombre':['Ana','Luis','Carlos']}
datos_2 = {'Id':[1,2,4],'Edad':[25,30,22]}''')
                
                datos_1 = {'Id':[1,2,3], 'Nombre':['Ana','Luis','Carlos']}
                datos_2 = {'Id':[1,2,4],'Edad':[25,30,22]}
                
                
                st.code('''df1 = pd.DataFrame(datos_1)
df1''')
                
                df1 = pd.DataFrame(datos_1)
                df1
                
                st.code('''df2 = pd.DataFrame(datos_2)
df2''')                
                
                df2 = pd.DataFrame(datos_2)
                df2



                st.write('inner: se genera unicamente con los datos que coinciden en ambos DataFrame.')
                
                st.code('''df_combinado = pd.merge(df1, df2, on='Id')
df_combinado''')                
                
                df_combinado = pd.merge(df1, df2, on='Id')
                df_combinado
                
  
                st.write('outer: combina todas las filas agregando None donde no encuentren los resultados.')
                st.code('''df_combinado = pd.merge(df1, df2, on='Id', how='outer')
df_combinado''')                    
                
                df_combinado = pd.merge(df1, df2, on='Id', how='outer')
                df_combinado  
                
                st.divider()
                st.write('''**join():**: permite unir dos DataFrames a partir de un indice o una columna clave .  
                ''') 
                
                st.code('''datos_1 = {'Salario':[30000,45000,38000], 'Antiguedad':[9,13,12]}
datos_2 = {'Ciudad':['Madrid', 'Barcelona', 'Valencia'], 'Jerarquia':['Baja','Media','Alta']}''')                
                
                datos_1 = {'Salario':[30000,45000,38000], 'Antiguedad':[9,13,12]}
                datos_2 = {'Ciudad':['Madrid', 'Barcelona', 'Valencia'], 'Jerarquia':['Baja','Media','Alta']}
                
                st.code('''df1 = pd.DataFrame(datos_1, index=[1,2,3])
df1''')
                
                df1 = pd.DataFrame(datos_1, index=[1,2,3])
                df1
                
                st.code('''df2 = pd.DataFrame(datos_2, index=[1,2,4])
df1''')
                df2 = pd.DataFrame(datos_2, index=[1,2,4])
                df2
                
                st.write('Por defecto join tiene el parametro how = left')
                
                st.code('''df_unido = df1.join(df2)
df_unido''')                
                df_unido = df1.join(df2)
                df_unido
                
                
                st.divider()
                
                st.write('''
                **concat():**: permite unir dos DataFrames a partir de un eje (vertical o horizontal) .  
                ''')         
                
                st.code('''datos_1 = {'Nombre':['Juan','Gabrieal','Elena']}
datos_2 = {'Nombre':['Carmela','Max','Laura']}''')                            
                
                datos_1 = {'Nombre':['Juan','Gabrieal','Elena']}
                datos_2 = {'Nombre':['Carmela','Max','Laura']}
                
                
                st.code('''df1 = pd.DataFrame(datos_1)
df1''')                 
                df1 = pd.DataFrame(datos_1)
                df1
                
                
                st.code('''df2 = pd.DataFrame(datos_2)
df2''')                     
                df2 = pd.DataFrame(datos_2)
                df2
                
                st.write('''Por defecto el concat es vertical, para que sea horizontal debe tener el parametro axis = 1.
keys: permite identificar a que DataFrame pertenece cada indice.        
ignore_index: si es True el indice del 2do DataFrame continua al del indice del DataFrame 1.''')
                
                st.code('''df_concatenado = pd.concat([df1, df2], keys=['df1', 'df2'])
df_concatenado''')                      
                
                df_concatenado = pd.concat([df1, df2], keys=['df1', 'df2'])
                df_concatenado
                
                
                st.code('''df_contatenado_h = pd.concat([df1, df2],keys=['df1', 'df2'], axis=1)
df_contatenado_h''')                 
                df_contatenado_h = pd.concat([df1, df2],keys=['df1', 'df2'], axis=1)
                df_contatenado_h
         

        if opcion_seleccionada == 'Respaldos':   
                
                st.code('''df_tips = pd.read_csv('DataFrames/tips.csv')''')
                df_tips = pd.read_csv('DataFrames/tips.csv')
                
                st.write('**copy()**: permite crear una copia independiente del DataFrame original guardandola en una nueva variable.')
                st.code('''df_backup = df_tips.copy()''')
                df_backup = df_tips.copy()
                
                
                st.write('**to_csv()**: permite exportar un DataFrame en formaro csv, usando comas como separador y sin incluir el indice del DataFrame en el archivo.')
                st.code('''def descargar_df_csv(dataframe, nombre_archivo):
        dataframe.to_csv(nombre_archivo, index=False)''')
                
                def descargar_df_csv(dataframe, nombre_archivo):
                        dataframe.to_csv(nombre_archivo, index=False)
                        
                st.write('Exportar fichero')
                st.code('''descargar_df_csv(df_backup, 'DataFrames/tips_bkp.csv')''')
                descargar_df_csv(df_backup, 'DataFrames/tips_bkp.csv')
                 

 
 
 
 
 

def matplotlib():

        opciones_plt = ['plot','scatter','hist','bar','boxplot','pie']
    
        # Carga de Datos
        df = pd.read_csv('DataFrames/tips.csv')
    
        col1, col2 = st.columns([2,2])
    
        with col1:
                opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_plt)
                st.success(f'##### **{opcion_seleccionada}** ')
      
    
        if opcion_seleccionada == 'plot':
                st.write('''Matplotlib grafica sus datos en Figuras, cada una de las cuales puede contener uno mas Ejes, una area donde los puntos se pueden especificar en terminos de coordenadas x-y.        
La forma mas sencilla de crear una figura con ejes es utilizar pyplot.subplots()''')                

                st.write('Función que genera gráficos de línea')
        

                codigo = '''x = np.linspace(0,5,11)     
y = x**2
z = x**3'''     
                st.code(codigo)
        
        
                x = np.linspace(0,5,11)
                y = x**2
                z = x**3
        
                st.write('##### Parámetros')
        
                st.write('''
        * figsize -> (ancho, alto)          
        * label -> determina el nombre de la etiqueta     
        * color -> color del grafico (nombre o codigo hexadecimal)    
        * linewidth o lw -> ancho de la linea
        * linestyle o ls -> tipo de linea (--)(-.)(:)(steps)
        * marker -> puntos de interseccion de x e y (o)(+)(*)(s)
        * markersize -> tamaño del marker
        * markerfacecolor -> color interior marker
        * markeredgewith -> tamano del borde del maker
        * markeredgecolor -> color del borde del maker
        * legend -> muestra leyenda con las etiquetas)
        * loc -> determina el lugar de legenda (0: best)
        * title -> titulo del grafico
        * xlabel -> etiqueta eje x
        * ylabel -> etiqueta eje y''')
        
        
                codigo = '''fig,ax = plt.subplots(figsize=(6,6))            
ax.plot(x,y,label='X Square', color='blue', linewidth=3, linestyle='--')
ax.plot(x,z,label='X Cubed', color='#ff8c00', linewidth=.8, marker='s', markersize=7, markerfacecolor='yellow', markeredgewidth=1, markeredgecolor='green')
ax.legend(loc=0)

plt.title('Grafico de lineas')
plt.xlabel("Eje X")
plt.ylabel("Eje Y")

st.pyplot(fig)'''
                st.code(codigo)
        
        
                # plot
                with st.container(width=800):
                        fig,ax = plt.subplots(figsize=(6,6))
                
                        ax.plot(x,y,label='X Square', color='blue', linewidth=3, linestyle='--')  
                        ax.plot(x,z,label='X Cubed', color='#ff8c00', linewidth=.8, marker='s', markersize=7, markerfacecolor='yellow', markeredgewidth=1, markeredgecolor='green')
                        ax.legend(loc=0)
                
                        plt.title('Grafico de lineas')
                        plt.xlabel("Eje X")
                        plt.ylabel("Eje Y")
                
                        st.pyplot(fig)   

        
        
                codigo = '''years = [1950,1960,1970,1980,1990,2000,2010]
gdp = [300.2,543.3,1075.9,2862.5,5979.6,10289.7,10958.3] 

fig,ax = plt.subplots(figsize=(6,6))
ax.plot(years, gdp, color='green', marker='o', linestyle='solid')

plt.title('Nominal GDP')
plt.ylabel('Millions U$S')  

st.pyplot(fig)'''
                st.code(codigo)

                with st.container(width=800):
                        years = [1950,1960,1970,1980,1990,2000,2010]
                        gdp = [300.2,543.3,1075.9,2862.5,5979.6,10289.7,10958.3]        
                        
                        fig,ax = plt.subplots(figsize=(6,6))
                        ax.plot(years, gdp, color='green', marker='o', linestyle='solid')
                        plt.title('Nominal GDP')
                        plt.ylabel('Millions U$S')  
                        
                        st.pyplot(fig)     
                
                
        if opcion_seleccionada == 'scatter':
                st.write('Función que se usa para crear diagramas de dispersión, que son gráficos que muestran la relación entre dos variables numéricas utilizando puntos en un plano cartesiano')
        
                codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
                st.code(codigo)
                st.dataframe(df.head())

                st.write('##### Parámetros')
                st.write('''
        * set_title -> Titulo del grafico          
        * st_xlabel -> Label eje x
        * st_ylabel -> Label eje y
        * s -> tamaño del marker
        * c -> color del marker
        * marker -> o (circulos), ^ (triangulos), * (estrellas)
        * alpha -> transparencia (0 a 1)
        * cmap -> colormap (viridis, plasma)
        ''')

                codigo = '''fig,ax = plt.subplots()   
ax.scatter(x=df['total_bill'], y=df['tip'], c='#ff8c00', s=5, alpha=.5)
ax.set_title('Diagrama de Dispersión (Total de la Cuenta vs Propina)')
ax.set_xlabel('Total de la cuenta')
ax.set_ylabel('Propina')

st.pyplot(fig)'''     
                st.code(codigo)

                # scatter
                with st.container(width=700):
                        fig,ax = plt.subplots()

                        ax.scatter(x=df['total_bill'], y=df['tip'], c='#ff8c00', s=5, alpha=.5)
                        ax.set_title('Diagrama de Dispersión (Total de la Cuenta vs Propina)')
                        ax.set_xlabel('Total de la cuenta')
                        ax.set_ylabel('Propina')
                        st.pyplot(fig)
            
        if opcion_seleccionada == 'hist':
                st.write('Función para crear histogramas y visualizar la distribución de datos numéricos, agrupándolos en intervalos (bins) y mostrando la frecuencia de los valores en cada uno.')
                
                codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
                
                st.code(codigo)
                st.dataframe(df.head())

                st.write('##### Parámetros')
                st.write('''
                * bins -> define el nro de columnas que se muestran en el grafico.
                * range -> tupla que define el rango inferior y superior de los contenedores (minimo, maximo). Los valores fuera de rango se ignoran. 
                * density -> si es True, el histograma se normaliza para que el area total sea 1.         
                * edgecolor -> establece el color de los bordes de las barras.  
                * color -> define el color de la barra.
                * histtype -> tipo histograma: bar (predeterminado), barstacked, step, stepfilled. 
                * orientation -> orientacion de las barras: vertical o horizontal.   
                * alpha -> determina la opacidad''')
                


                codigo = '''fig,ax = plt.subplots()   
ax.hist(x=df['total_bill'], bins=15, edgecolor='#000000', color='#8b92cc', alpha=.8)
ax.set_title('Histograma (Distribución del Total de la Cuenta)')
ax.set_xlabel('Total de la Cuenta')
ax.set_ylabel('Frecuencia')

st.pyplot(fig)'''     
                st.code(codigo)

                # hist
                with st.container(width=700):
                        fig,ax = plt.subplots()

                        ax.hist(x=df['total_bill'], bins=15, edgecolor='#000000', color='#8b92cc', alpha=.8)
                        ax.set_title('Histograma (Distribución del Total de la Cuenta)')
                        ax.set_xlabel('Total de la Cuenta')
                        ax.set_ylabel('Frecuencia')
                        st.pyplot(fig)
            
        if opcion_seleccionada == 'bar':
                st.write('Función para crear gráficos de barra.')
                
                codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
                st.code(codigo)
                st.dataframe(df.head())        
                
                st.write('##### Parámetros')
                
                st.write('''
                * edgecolor -> color borde de la barra.
                * height -> altura de la barra.        
                * width -> ancho de la barra.  
                * bottom -> coordenadas y donde inician las barras, valor predeterminado 0.
                * align -> alineacion de la barra respecto a la coordenada x: center (centrado) o edge (borde izquierdo).
                * xticks() -> determina las etiquetas en el eje x
                * axis -> rango de valores en el eje x e y (inicio-fin)
''')         
            
            
                codigo = '''fig,ax = plt.subplots()   
ax.bar(df['day'],df['total_bill'])
ax.set_title('Total de la Cuenta por día')
ax.set_xlabel('Día')
ax.set_ylabel('Total de la Cuenta')

st.pyplot(fig)'''     
                st.code(codigo)

        
                with st.container(width=700):
                        fig,ax = plt.subplots()

                        ax.bar(df['day'],df['total_bill'])
                        ax.set_title('Total de la Cuenta por día')
                        ax.set_xlabel('Día')
                        ax.set_ylabel('Total de la Cuenta')
                        st.pyplot(fig)            
        
        
                codigo = '''movies = ['Annie Hall','Ben-Hur','Casablanca','Gandhi','West Side Story']
num_oscars = [5,11,3,8,10]        
        
fig,ax = plt.subplots()   
ax.bar(movies, num_oscars,edgecolor='black',width=.5)
plt.title('Oscars por pelicula')
plt.ylabel('Nro de Premios')

st.pyplot(fig)'''     
                st.code(codigo)

                movies = ['Annie Hall','Ben-Hur','Casablanca','Gandhi','West Side Story']
                num_oscars = [5,11,3,8,10]
                
                with st.container(width=700):
                        fig,ax = plt.subplots()

                        ax.bar(movies, num_oscars,edgecolor='black',width=.5)
                        ax.set_title('Oscars por pelicula')

                        ax.set_ylabel('Nro de Premios')
                        st.pyplot(fig)  
        
        
                codigo = '''mentions = [500,505]
years = [2017,2018]       
        
fig,ax = plt.subplots()   
ax.bar(years, mentions)
ax.set_xticks(years)
ax.axis([2016.5,2018.5,499,506])

st.pyplot(fig)'''     
                st.code(codigo)

                mentions = [500,505]
                years = [2017,2018]
                
                with st.container(width=700):
                        fig,ax = plt.subplots()

                        ax.bar(years, mentions,width=.6)
                        ax.set_xticks(years)
                        ax.axis([2016.5,2018.5,499,506])
                        st.pyplot(fig)  

 
        if opcion_seleccionada == 'boxplot':
                st.write('Función para crear diagramas de caja.')
                
                codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
                st.code(codigo)
                st.dataframe(df.head())              
            
                st.write('##### Parámetros')
                
                st.write('''
                * notch -> crea una muesca si es True en la mediana para mostrar el intervalo de confianza.
                * vert -> cambia la orientacion del grafico de vertical a horizontal, si es False.
                * patch_artist -> rellena las cajas con color si es True.
                * showmeans -> muestra la media aritmetica con un marcador adicional si es True.
                * showfliers -> oculta los valores atipicos si es False.
                * labels -> lista de strings para etiquetas de cada caja.       
                * width -> ancho de la caja.  
                * boxprops -> estilo de la caja (color de borde, relleno).
                * medianprops -> estilo de la linea de la mediana.
                * whiskerprops -> estilo de los bigotes (lineas).
                * capprops -> estilo de los 'gorros' en los extremos de los bigotes.
                * flierprops -> estilo de los valores atipicos (puntos). 
''')         
                   
 
 
                codigo = '''outlier_style = dict(marker='o', markerfacecolor='red', markersize=6, markeredgecolor='black')
                
fig,ax = plt.subplots()   
ax.boxplot(df['total_bill'], showmeans=True, flierprops=outlier_style)
ax.set_title('Total de la Cuenta por día')
ax.set_ylabel('Total de la Cuenta')

st.pyplot(fig)'''     
                st.code(codigo)

                outlier_style = dict(marker='o', markerfacecolor='red', markersize=6, markeredgecolor='black')
        
                with st.container(width=800):
                        fig,ax = plt.subplots()

                        ax.boxplot(df['total_bill'], showmeans=True, flierprops=outlier_style)
                        ax.set_title('Total de la Cuenta por día')
                        ax.set_ylabel('Total de la Cuenta')
                        st.pyplot(fig)     

 
        if opcion_seleccionada == 'pie':
                st.write('Función para crear gráficos circulares.')
                
                codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
                st.code(codigo)
                st.dataframe(df.head())  
        
                st.write('##### Parámetros')
                st.write('''
                * labels -> lista de las cadenas para etiquetar cada sector.
                * explode -> lista que especifica la fraccion del radio con la que se desplaza cada segmento del centro.
                * autopct -> cadena de formato para añadir porcentajes dentro de las cuñas.
                * startangle -> angulo en grados que rota el inicio de la primera porcion desde el eje X.
                * shadow -> valor booleano para añadir sombra bajo el grafico.
                * radius -> radio del grafico (1 por defecto).
                * labeldistance -> distancia radial a la que se dibujan las etiquetas.
                * pctdistance -> relacion entre el centro y el inicio del texto generado por autopct.
                * counterclock -> booleano para especificar la direccion horaria o antihoraria.
                * wedgeprops -> diccionario para personalizar el estilo de las cuñas (borde, grosor).
                * textprops -> diccionario para personalizar el estilo del texto (color, tamaño). 
                * colors -> colores''') 
 
 
                codigo = '''colores = ['#87ceeb', '#6095']
value = df['sex'].value_counts()       

fig,ax = plt.subplots()   
ax.pie(value,labels=value.index,autopct='%0.2f%%')
ax.set_title('Total de registros por sexo')

st.pyplot(fig)'''     
                st.code(codigo)

        
                with st.container(width=800):
                        colores = ['#87ceeb', '#6095ed']
                        fig,ax = plt.subplots()
                        value = df['sex'].value_counts()
                        
                        ax.pie(value,labels=value.index,autopct='%0.2f%%', colors=colores)
                        ax.set_title('Total de registros por sexo')
                        st.pyplot(fig)     
 
 
 
def seaborn():
    
        opciones_sns = ['relplot','scatterplot','lineplot','displot','histplot','kdeplot', 'boxplot','barplot','violinplot','catplot','heatmap','pairplot']
    
        # Carga de Datos
        df = pd.read_csv('DataFrames/tips.csv')
    
        col1, col2 = st.columns([2,2])
    
        with col1:
                opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_sns)
                st.success(f'##### **{opcion_seleccionada}** ')
    
    
        if opcion_seleccionada == 'relplot':
                st.write('Grafico relacional, se utiliza para comprender la relacion entre dos variables numericas.')    

                st.code('''import seaborn as sns
df = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
df.head()''')
                st.dataframe(df.head())
        
                st.write('##### Tipos de Graficos')
                st.write('''**kind** = 'scatter': crea un grafico de puntos (predeterminado), 'line': crea un grafico de lineas.
                         
                         ''')
        
                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes.    
        * hue -> colorea los puntos/lineas basado en una variable categorica.
        * col, row -> divide el grafico en subgraficos basados en variables.
        * size -> modifica el tamaño de los puntos segun una variable numerica.
''')        
        
                st.code('''fig, ax = plt.subplots()
fig = sns.relplot(data=df,x='total_bill',y='tip',hue='sex')

st.pyplot(graf)''')        
        
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        fig = sns.relplot(data=df,x='total_bill',y='tip',hue='sex')
        
                        st.pyplot(fig) 
        
        
        if opcion_seleccionada == 'scatterplot':
                st.write('Grafico relacional, de dispersion para ver la relacion entre variables.')    
        
                st.code('''import seaborn as sns
df = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
df.head()''')
                st.dataframe(df.head())        
        
                st.write('##### Parámetros')
        
                st.write('''
        * hue -> colorea los puntos por una variable categorica.      
        * size -> modifica el tamaño de los puntos segun una variable numerica.
        * style -> cambiar la forma del marcador segun una variable.
''')        
        
                st.code('''fig, ax = plt.subplots()
sns.scatterplot(data=df,x='total_bill',y='tip',hue='time',style='sex')

st.pyplot(fig) ''')        
        
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.scatterplot(data=df,x='total_bill',y='tip',hue='time',style='sex')
        
                        st.pyplot(fig) 


        if opcion_seleccionada == 'lineplot':
                st.write('Grafico relacional, de lineas para tendencias en el tiempo.')    
        
        
                st.code('''import seaborn as sns
df = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
df.head()''')
                st.dataframe(df.head())        
        
                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes.    
        * hue -> colorea los puntos/lineas basado en una variable categorica.
        * style -> cambiar la forma del marcador segun una variable.
        * size -> modifica el grosor de la linea segun una variable.
        * markers -> añade marcadores a los puntos de datos si es True.
''')        
        
                st.code('''fig, ax = plt.subplots()
sns.lineplot(data=df,x='day',y='total_bill')

st.pyplot(fig) ''')        
        
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.lineplot(data=df,x='day',y='total_bill')
        
                        st.pyplot(fig) 

        
        if opcion_seleccionada == 'displot':
                st.write('Gráfico de distribucion, examina distribuciones univariantes o bivariantes.')
        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * kind -> tipo de grafico: hist (predeterminado), kde, ecdf.
        * kde -> añade una curva de estimacion de densidad de kernel sobrel el histograma si es True.
        * hue -> define subconjuntos de datos por color.    
          
''')
        
                st.code('''fig, ax = plt.subplots()
fig = sns.displot(data = df, x='total_bill', kde=True, hue='sex')

st.pyplot(graf)''')

                # displot
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        fig = sns.displot(data = df, x='total_bill', kde=True, hue='sex')
            
                        st.pyplot(fig)  
 


        if opcion_seleccionada == 'histplot':
                st.write('Gráfico de distribucion, histogramas para visualizar la distribucion de una variable.')
        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes.    
        * bins -> define el numero de barras o intervalos.
        * kde -> para dibujar una curva de estimacion de densidad si es True.
        * hue -> variable semantica para colorear grupos distintos.
        * stat -> tipo de estadistica: count, frequency, density o probability.
''')
        
                st.code('''fig, ax = plt.subplots()
sns.histplot(data=df, x='total_bill', bins=12, stat='frequency', hue='time', kde=True)

st.pyplot(graf)''')

                # histplot
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.histplot(data=df, x='total_bill', bins=12, stat='frequency', hue='time', kde=True)
            
                        st.pyplot(fig)  
 

        if opcion_seleccionada == 'kdeplot':
                st.write('Gráfico de distribucion, de densidad de nucleo.')
        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes.    
        * fill -> rellena el area bajo la curva si es True.
        * hue -> divide las observaciones por una variable categorica para comparar grupos.
        * multiple -> define como graficar multiples distribuciones (layer, stack, fill, dodge) 
''')
        
                st.code('''fig, ax = plt.subplots()
sns.kdeplot(data=df, x='total_bill', fill=True, hue='sex', multiple='layer')

st.pyplot(graf)''')

                # kdeplot
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.kdeplot(data=df, x='total_bill', fill=True, hue='sex', multiple='layer')
            
                        st.pyplot(fig)  



        if opcion_seleccionada == 'boxplot':
                st.write('Gráfico categorico, diagrama de caja para visualizar distribuciones, medianas y valores atipicos (outliers).')
                st.write('''La caja (box) representa el rango intercuartilico (IQR), donde se encuentra el 50% central de los datos (entre cuartil 1 y el cuartil 3).   
La linea central es la mediana de los datos.    
Los bigotes (whiskers) son las lineas que se extienden desde la caja para mostrar la variabilidad fuera del 50% central, generalmente cubriendo hasta 1.5xIQR.  
Los valores atipicos (outliers) son puntos individuales mas alla de los bigotes, considerados inusuales.''')
        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes, x categorica, y valor.    
        * hue -> divide las observaciones por una variable categorica para comparar grupos.
        * order / hue_order -> listas de strings para controlar el orden de las categorias en el eje.
        * orient -> orientacion del grafico (v para vertical, h para horizontal).
        * palette -> paleta de colores para definir los colores de las cajas.
        * whis -> define la longitud de los bigotes, el valor por defecto es 1.5 (corresponde a 1.5 x IQR). 
        * with -> ancho de las cajas. 
        * showmeans -> para mostrar la media con un punto o linea adicional si es True.
''')
        
                st.code('''fig, ax = plt.subplots()
sns.boxplot(data=df, y='total_bill', width=.2, hue='smoker')

st.pyplot(graf)''')


                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.boxplot(data=df, y='total_bill', width=.2, hue='smoker')
                        st.pyplot(fig)  

                st.code('''fig, ax = plt.subplots()
sns.boxplot(data=df, x='day', y='total_bill', palette='Set2')

st.pyplot(graf)''')

                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.boxplot(data=df, x='day', y='total_bill', palette='Set2')
            
                        st.pyplot(fig)  


        if opcion_seleccionada == 'barplot':
                st.write('Gráfico categorico, grafico de barras para comparar valores promedio entre categorias.')
        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes.  
        * hue -> divide las observaciones por una variable categorica para comparar grupos.
        * estimator -> funcion estadistica para estimar dentro de cada categoria. Por defecot es la media (mean), puede ser sum, median o funciones personalizadas.
        * errobar -> controla la visualizacion de la incertidumbre, sd para la desviacion estandar o None para desactivarla.
        * order / hue_order -> listas de strings para controlar el orden de las categorias en el eje.
        * orient -> orientacion del grafico (v para vertical, h para horizontal).
        * palette -> mapa de colores para las barras (viridis, pastel)
        * capsize -> ancho de los bigotes en las barras de error.
        * alpha -> transparencia de las barras (0 a 1).
''')
        
                st.code('''fig, ax = plt.subplots()
sns.barplot(data=df, x='day', y='total_bill', hue='sex', palette='viridis')

st.pyplot(graf)''')


                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.barplot(data=df, x='day', y='total_bill', hue='sex', palette='viridis')
                        st.pyplot(fig)  


        if opcion_seleccionada == 'violinplot':
                st.write('Gráfico categorico, combina aspectos de boxplot y la densidad (kde)')

        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes, x categorico, y numerico.  
        * hue -> divide las observaciones por una variable categorica para comparar grupos.
        * order / hue_order -> listas de strings para controlar el orden de las categorias en el eje.
        * split -> divide el violin en dos mitades (una por color) en lugar de mostrarlos por separado si es True.
        * scale -> define como se escala el ancho de cada violin (area, count, width).
        * inner -> representacion interna del violin (box, quartile, point, stick, None).
        * bw -> metodo para el calculo del ancho de banda (bandwidth), ajusta la suavidad de la densidad (scott, silverman o un numero).
        * palette -> colores a utilizar para los diferentes niveles de la variable hue.
        * linewidth -> grosor de las lineas del borde del violin.
''')
        
                st.code('''fig, ax = plt.subplots()
sns.violinplot(data=df, x='day', y='total_bill', hue='smoker', palette='muted', split=True)

st.pyplot(graf)''')


                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.violinplot(data=df, x='day', y='total_bill', hue='smoker', palette='muted', split=True)
                        st.pyplot(fig) 


        if opcion_seleccionada == 'catplot':
                st.write('Gráfico categorico, figura de nivel superior para organizar graficos categoricos en facetas.')

        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.      
        * x,y -> nombres de las columnas para los ejes.  
        * kind -> tipo de grafico: strip (defecto), swarm, box, violin, boxen, point, bar o count.
        * hue -> divide las observaciones por una variable categorica para comparar grupos.
        * col, row -> variables categoricas para crear subgraficas (facets) organizadas en columnas o filas.
        * palette -> mapa de colores para las barras (viridis, pastel).
        * height / aspect -> tamaño de la figura (altura) y relacion de ancho-alto.
        * order / hue_order -> listas de strings para controlar el orden de las categorias en el eje.
''')
        
                st.code('''fig, ax = plt.subplots()
fig = sns.catplot(data=df, x='day', y='total_bill', kind='point', hue='sex')

st.pyplot(graf)''')


                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        fig = sns.catplot(data=df, x='day', y='total_bill', kind='point', hue='sex')
                        st.pyplot(fig) 


        if opcion_seleccionada == 'heatmap':
                st.write('Mapa de calor, util para visualizar matrices de correlacion.')

                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> conjunto de datos 2D (ndarray, DataFrame).
        * annot -> muestra el valor numerico en cada celda si es True.
        * annot_kws -> diccionario para modificar fuente de annot (size, color).
        * fmt -> formato de cadena para las anotaciones (ej '.2f' para flotantes, 'd' para enteros).
        * square -> fuerza que las celdas sean cuadradas si es True. 
        * cmap -> mapa de colores (viridis, coolwarm, blues). 
        * vmin, vmax -> valores para fijar los limites del mapa de color.
        * center -> valor en el que se centra el mapa de colores al visualizar datos divergentes.
        * cbar -> oculta la barra de color si es False.
        * linewidths -> ancho de lina entre celdas.
        * linecolor -> color de las lineas divisorias.
        * xticklabels, yticklabels -> controla la visualizacion de las etiquetas de los ejes, si es False se oculta.    

''')
        
                st.write('###### Conversion de datos categoricos')
                
                st.code('''sex = {'Female':0, 'Male':1}
smoker = {'No':0, 'Yes':1}
day = {'Sun':0, 'Sat':1, 'Thur':2, 'Fri':3}
time = {'Lunch':0, 'Dinner':1}  

df_mod = df.copy()
df_mod['sex'] = df_mod['sex'].replace(sex)     
df_mod['smoker'] = df_mod['smoker'].replace(smoker)  
df_mod['day'] = df_mod['day'].replace(day)  
df_mod['time'] = df_mod['time'].replace(time)         

df_mod.head(10)''')
                
                sex = {'Female':0, 'Male':1}
                smoker = {'No':0, 'Yes':1}
                day = {'Sun':0, 'Sat':1, 'Thur':2, 'Fri':3}
                time = {'Lunch':0, 'Dinner':1}

                df_mod = df.copy()
                df_mod['sex'] = df_mod['sex'].replace(sex)     
                df_mod['smoker'] = df_mod['smoker'].replace(smoker)  
                df_mod['day'] = df_mod['day'].replace(day)  
                df_mod['time'] = df_mod['time'].replace(time)     
                        
                st.code(df_mod.head(10), language='html')           
        
        
                st.code('''matriz_correlacion = df_mod.corr()
                           
fig, ax = plt.subplots()
sns.heatmap(data=matriz_correlacion, annot=True, cmap='viridis', linecolor='white', linewidths=.4, annot_kws={'size':7, 'color':'white'})

st.pyplot(graf)''')

     
                        
                matriz_correlacion = df_mod.corr()
                st.code(matriz_correlacion)
                
                with st.container(border=True, width=800):
                        fig, ax = plt.subplots()
                        sns.heatmap(data=matriz_correlacion, annot=True, cmap='viridis', linecolor='white', linewidths=.4, annot_kws={'size':7, 'color':'white'})
                        st.pyplot(fig) 



        if opcion_seleccionada == 'pairplot':
                st.write('Crea una matriz de graficos de dispersion para todas las relaciones de pares en un conjunto de datos.')

        
                st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
                st.dataframe(df.head())

                st.write('##### Parámetros')
        
                st.write('''
        * data -> DataFrame de pandas.  
        * hue -> divide las observaciones por una variable categorica para comparar grupos.    
        * vars -> lista de nombres de variables para incluir, en lugar de usar todas las numericas. 
        * kind -> tipo de grafico para las celdas fuera de la diagonal (scatter, kde, hist o reg). 
        * diag_kind -> tipo de grafico para la diagonal principal (auto, hist, kde o None para ocultar). 
        * palette -> mapa de colores para las barras (viridis, pastel)        
        * height -> define la altura (en pulgadas) de cada subgrafico. 
        * corner -> solo muestra la esquina infeior izquierda de la matruz si es True. 
        * markers -> personaliza los marcadores para cada grupo de variables hue. 
        * plot_kws y diag_kws -> diccionarios para pasar argumentos adicionales a los graficos de dispersion y diagonales, respectivamente.
''')
        
                st.code('''fig, ax = plt.subplots()
fig = sns.pairplot(data=df, diag_kind='kde', hue='sex')

st.pyplot(fig) ''')


                with st.container(border=True, width=1000):
                        fig, ax = plt.subplots()
                        fig = sns.pairplot(data=df, diag_kind='kde', hue='sex')
                        st.pyplot(fig) 

 
            
def git():

    opciones_git = ['Git', 'GitHub']
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_git)
        st.success(f'##### **{opcion_seleccionada}** ')
    

    if opcion_seleccionada == 'Git':
        st.write('''**Definción:** es un sistema de Control de Versiones.    
GIT permite compartir el codigo con otras personas. Existe un concepto que se llama Deployment, la idea es que cuando el proyecto este finalizado y se quiera llevar a produccion, se pueda hacer el deploy en la nube.    

**Sistema de control de versiones:** es un conjunto de prodcedimientos que registra los cambios en una archivo
en un conjunto de archivos a lo largo del tiempo de modo que se pueda recuperar versiones especificas mas adelante.
- Regresar a versiones anteriores de los archivos.
- Regresar a versiones anteriores del proyecto completo.
- Comparar cambios a lo largo del tiempo.
- Ver quien modifico un archivo en un momento especifico.
- Recuperar archivos perdidos o arruinados.            
''')



def aws():
    st.write('##### Definición')
    st.write('''Amazon Web Services es una colección de servicios de computación en la nube pública que en conjunto forman una plataforma de computación en la nube, 
ofrecidas a través de Internet por Amazon.com. Es usado en aplicaciones populares como Dropbox, Foursquare, HootSuite.''')

def docker():
    st.write('##### Definición')
    st.write('''Es una plataforma de código abierto que permite a los desarrolladores crear, desplegar y ejecutar aplicaciones en contenedores,     
que son unidades estandarizadas y aisladas que incluyen todo lo necesario para que el software funcione, como el código, las bibliotecas y las dependencias.    
Su uso principal es estandarizar y automatizar el ciclo de vida del desarrollo y la implementación de software, garantizando que una aplicación se ejecute de la misma
manera en cualquier entorno.

Docker crea un conjunto de herramientas para que se pueda extender, normalizar y generalizar el uso e los contenedores.     
Así, a través de Docker, podemos crear, desplegar y ejecutar aplicaciones mediante el uso e contenedores.

Docker permite a los desarrolladores y operadores empaquetar aplicaciones dentro de estos contenedores y a la vez intgrar este proceso a Pipeline.

**CI (Continuous Integration)**: es una de las metodologías de software muy ligada a la metoología ágil que nos permite una integración continuna y un despligue continuo.  
**CD (Continuous Delivery)**: liberación automática al repositorio de código.   
**CD (Continuous Deployment)**: despliegue automático hacia producción.
''')
    
    st.write('---')
    st.write('##### Instalación')
    st.write('''**Web descarga**: https://docs.docker.com/   
Docker Desktop para Windows: https://docs.docker.com/desktop/setup/install/windows-install/
         
''')
    
    imagen = Image.open('Imagenes/docker_1.png')
    st.image(imagen, width=919)        
    
    st.write('---')
    st.write('##### Ejecución Imagen de Docker')
    st.write('''**Imagenes**: https://hub.docker.com/
''')
    
    imagen = Image.open('Imagenes/docker_2.png')
    st.image(imagen, width=1826)       
    st.write('')
    st.write('**Seleccion de Imagenes postgres**')
    imagen = Image.open('Imagenes/docker_3.png')
    st.image(imagen, width=1150)     
    
    st.write('---')    
    st.write('##### Descargar Imagen (latest)')
    
    st.code('docker run postgres',language='cmd')
    
    imagen = Image.open('Imagenes/docker_4.png')
    st.image(imagen, width=1133)          
    imagen = Image.open('Imagenes/docker_5.png')
    st.image(imagen, width=1254)          

    st.code('docker run -e POSTGRES_PASSWORD=password postgres',language='cmd')
    imagen = Image.open('Imagenes/docker_6.png')
    st.image(imagen, width=1143)       

    st.write('docker images muestra las imagenes que estan en la PC')
    st.code('docker images',language='cmd')
    imagen = Image.open('Imagenes/docker_7.png')
    st.image(imagen, width=1104)      

    st.write('docker ps muestra los contenedores que estén en marcha.')
    st.code('docker ps',language='cmd')
    imagen = Image.open('Imagenes/docker_8.png')
    st.image(imagen, width=1171)      

    st.write('---')
    st.write('##### Descargar Imagen con otro TAG')
    imagen = Image.open('Imagenes/docker_9.png')
    st.image(imagen, width=1342)   

    st.code('docker run -d -e POSTGRES_PASSWORD=password postgres:bookworm')
    
    imagen = Image.open('Imagenes/docker_10.png')
    st.image(imagen, width=1342)   
    
    st.write('Se descargan dos imagenes (versiones de Postgres diferentes).')
    
    
    imagen = Image.open('Imagenes/docker_11.png')
    st.image(imagen, width=1218)       
    
    st.write('---')
    st.write('##### Crear contenedor con nombre personalizado')
    st.code('docker run -d -e POSTGRES_PASSWORD=password --name bookworm_container postgres:bookworm')
    imagen = Image.open('Imagenes/docker_12.png')
    st.image(imagen, width=1232)     

    st.write('---')
    st.write('##### Eliminar contenedores')
    st.code('''# Contenedores en marcha y detenidos     
docker ps -a''', language='html')
    imagen = Image.open('Imagenes/docker_13.png')
    st.image(imagen, width=1232)  
    
    st.code('''# Eliminar un contenedor inactivo por su Container ID  o Name 
docker rm ba64470c600d''', language='html')
  
    st.code('''# Eliminar un contenedor activo por su Container ID o Name. Para poder eliminarlo se debe forzar mediante el parámetro -f  
docker rm -f ba64470c600d quizzical_chandrasekhar''', language='html')
    imagen = Image.open('Imagenes/docker_14.png')
    st.image(imagen, width=1232)     

    st.write('---')
    st.write('##### Eliminar imagenes')

if __name__ == '__main__':
    main()

