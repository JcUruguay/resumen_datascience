import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import metrics
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    opciones_2 = ['Seleccionar...', 'Modelado de Datos','Regresión Lineal','Regresión Logística']
    
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
    
    
    st.sidebar.info('Ultima Actualización: 18/11/2025')
    
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
        
        if opcion_seleccionada_2 == 'Modelado de Datos':
            ml_modelado()        
        elif opcion_seleccionada_2 == 'Regresión Lineal': 
            ml_regresion_lineal()
        elif opcion_seleccionada_2 == 'Regresión Logística':
            ml_regresion_logistica()
          
            
    if opcion_seleccionada_3 != 'Seleccionar...':
        st.title(opcion_seleccionada_3)
        
        if opcion_seleccionada_3 == 'Git y GitHub':
            git()
        
        
        
def ml_modelado():
    opciones_mlmodleado = ['tips']
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar Data Frame: ', opciones_mlmodleado)
        st.success(f'##### **{opcion_seleccionada}** ')
    

    if opcion_seleccionada == 'tips':
        st.write('##### Data Frame que muestra las Propinas de un Restaurante')
        
        #codigo
        codigo = '''
df = pd.read_csv('DataFrames/tips.csv')
st.dataframe(df.head(10))'''
        st.code(codigo)
        

        
        # Carga de Datos
        df = pd.read_csv('DataFrames/tips.csv')
        
        # Visualizacion de 10 primeros registros
        st.write('##### Visualización de los primeros 10 registros')
        st.dataframe(df.head(10))
        
        st.write('---')
        
        st.write('##### Pie Chart y Bar Chart con Matplotlib')
        
        #codigo
        codigo = '''
data_types = df.dtypes      # Obtiene los tipos de datos del dataframe (int, float, object)
cat_cols = tuple(data_types[data_types == 'object'].index)      # Devuelve tupla con las columnas de tipo object
feature = st.selectbox('Seleccionar categoría', cat_cols, width=400)    # Seleccion de la columna
value = df[feature].value_counts()      # DataFrame con la cantidad por tipo

st.write('Pie Chart')
fig,ax = plt.subplots()
ax.pie(value,labels=value.index,autopct='%0.2f%%')
st.pyplot(fig)

st.write('Bar Chart')
fig,ax = plt.subplots()
ax.bar(value.index,value)       
st.pyplot(fig)

st.dataframe(value)'''
        st.code(codigo)        
        
        
        # Crear contenedor para seleccion de categoria
        with st.container(border=True):
            
            data_types = df.dtypes
            cat_cols = tuple(data_types[data_types == 'object'].index)
            
            feature = st.selectbox('Seleccionar categoría', cat_cols, width=400)

            value = df[feature].value_counts()
            col_1, col_2 = st.columns(2)
            
            
            with col_1:
                st.write('Pie Chart')
                # pie chart
                fig,ax = plt.subplots()
                ax.pie(value,
                    labels=value.index,
                    autopct='%0.2f%%')
                
                st.pyplot(fig)
            
            with col_2:
                # bar char
                st.write('Bar Chart')
                fig,ax = plt.subplots()
                ax.bar(value.index,value)
                
                st.pyplot(fig)
                 
            with st.expander(f'Cantidad por {feature}', width=400):
                    st.dataframe(value)        
        
        
        st.write('---')
        
        st.write('##### Graficos Box, Violin, Kdeplot y Hisplot con Seaborn')
        
        #codigo
        codigo = '''
grafico = st.selectbox('Seleccionar tipo de gráfico', ('Box','Violin','Kdeplot','Histogram'), width=300)
fig,ax = plt.subplots()
if grafico == 'Box':
    sns.boxplot(x='sex',y='total_bill', hue='sex', data=df)
elif grafico == 'Violin':
    sns.violinplot(x='sex', y='total_bill', hue='sex', data=df)
elif grafico == 'Kdeplot':
    sns.kdeplot(data=df, x='total_bill', hue='sex', fill=True)
elif grafico == 'Histogram':
    sns.histplot(x='total_bill', hue='sex',data=df)
    
st.pyplot(fig)'''
        st.code(codigo)            
        
        
        with st.container(border=True, width=1200):
            st.write('Distribución de Total Gastado por sexo')            
    
            # box, violin, kdeplot, histogram
            grafico = st.selectbox('Seleccionar tipo de gráfico', ('Box','Violin','Kdeplot','Histogram'), width=300)
            
            fig,ax = plt.subplots()
            if grafico == 'Box':
                sns.boxplot(x='sex',y='total_bill', hue='sex', data=df)
            elif grafico == 'Violin':
                sns.violinplot(x='sex', y='total_bill', hue='sex', data=df)
            elif grafico == 'Kdeplot':
                sns.kdeplot(data=df, x='total_bill', hue='sex', fill=True)
            elif grafico == 'Histogram':
                sns.histplot(x='total_bill', hue='sex',data=df)
            
            st.pyplot(fig)
                
                
        st.write('---')
        
        st.write('##### Grafico Scatterplot con Seaborn')
        
        #codigo
        codigo = '''
fig,ax = plt.subplots() 
hue_type = st.selectbox('Seleccionar categoría', cat_cols, width=300)
sns.scatterplot(x='total_bill',y='tip',hue=hue_type,data=df)   
 
st.pyplot(fig)
'''
        st.code(codigo)                  
                
                
        with st.container(border=True, width=1200):
            st.write('Grafico Scatter Total Gastado vs Propina')
            
            fig,ax = plt.subplots()
            hue_type = st.selectbox('Seleccionar categoría', cat_cols, width=300)
            
            sns.scatterplot(x='total_bill',y='tip',hue=hue_type,data=df)
            st.pyplot(fig)


        st.write('---')
        
        st.write('##### Gráfico Histograma con Plotly ')
        
        
        #codigo
        codigo = '''
fig,ax = plt.subplots() 
hue_type = st.selectbox('Seleccionar categoría', cat_cols, width=300)
sns.scatterplot(x='total_bill',y='tip',hue=hue_type,data=df)   
 
st.pyplot(fig)
'''
        st.code(codigo)                  
                        
        
        
        with st.container(border=True, width=1200):
            st.write('Grafico Histograma Total Gastado')
            
            # histogram (total bill)
            fig = px.histogram(df, x='total_bill', width=800)
            st.plotly_chart(fig)

            # histogram (total bill y color por sexo)
            fig = px.histogram(df, x='total_bill', color='sex', width=800)
            st.plotly_chart(fig)  



def ml_regresion_logistica():
    st.write('#### Definición')
    st.write('''La regresión logística es un método estadístico para predecir la probabilidad de un resultado categórico, como \'si\' o \'no\'.     
Utiliza una función logística (curva sigmoidea) para modelar la relación entre las variables independientes y una variable dependiente binaria (que solo tiene dos resultados).
Este modelo es útil en campos como la medicina para predecir la probabilidad de enfermedades o en la economía para predecir resultados de acciones.''')
    st.write('---')
    
    opciones_mlreglog = ['Opcion']

    col1, col2= st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlreglog)
        
    if opcion_seleccionada == 'Opcion':
        st.write('##### Data Frame')                                        

def ml_regresion_lineal():
    
    buffer = io.StringIO()
    
    st.write('#### Definición')
    st.write('''
La regresión lineal es un método estadístico que trata de modelar la relación entre una variable continua y una o más variables independientes mediante el ajuste de una ecuación lineal.   
Se llama regresión lineal simple cuando solo hay una variable independiente y regresión lineal múltiple cuando hay más de una.  
Dependiendo del contexto, a la variable modelada se le conoce como variable dependiente o variable respuesta, y a las variables independientes como regresores, predictores o features.''')
    st.write('---')
    
    opciones_mlreglin = ['USA Housing', 'Ecommerce Customers']
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_mlreglin)
        st.success(f'##### **{opcion_seleccionada}** ')
    
        @st.cache_data
        def load_data():
            # Carga del dataframe
            df = pd.read_csv('DataFrames/Casas.csv')
            
            return df
    
    # Precios de casa en Estados Unidos
    if opcion_seleccionada == 'USA Housing':
        st.write('##### Data Frame con los precios de casas en Estados Unidos')
        
        codigo = '''# Carga de csv con los datos
df = pd.read_csv('DataFrames/USA_housing.csv')
st.dataframe(df.head(10))       # head'''
        st.code(codigo)
        
        df = load_data()    
        st.dataframe(df.head(10))       # head
        codigo = '''# info
df = pd.read_csv('DataFrames/USA_housing.csv')
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
df = pd.read_csv('DataFrames/Ecommerce Customers')
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
        st.write('Función que se utiliza para mostrar mensajes en pantalla.')
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

        st.write('---')
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
 
        st.write('---')
        st.write('**shuffle()** reordena los items de una lista de forma aleatoria') 
        imagen = Image.open('Imagenes/random_2.png')
        st.image(imagen, width=1469)
 
        st.write('---')
        st.write('**choice()** retorna un elemento de una lista') 
        imagen = Image.open('Imagenes/random_3.png')
        st.image(imagen, width=1469)
 
 
        st.write('---')
        st.write('##### Referencias externas')       
     
    if opcion_seleccionada == 'Módulo statistics':      
        st.write('**Métodos:** mean, median, mode, stdev, pstdev, variance, pvariance')
        imagen = Image.open('Imagenes/statistics.png')
        st.image(imagen, width=1467)
 
        st.write('---')
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
    opciones_pd = ['Series','Data Frames','Exploración de Datos','Conversión de Tipos','Limpieza y Manipulación de Datos','Fusionar, Combinar y Concatenar Data Frames',
                    'Respaldos','Reporte ydata-profiling']
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_pd)
        st.success(f'##### **{opcion_seleccionada}** ')

    if opcion_seleccionada == 'Series':         
        st.write('''Una Serie es una estructura de datos unidimensional que puede contener cualquier tipo de datos.
    Es como una columna de una tabla.
    ''')
        
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
    Se compone de filas y columnas, donde cada columna puede contener un tipo de dato diferente.
    ''')
        st.write('##### Creación de un DataFrame')
        imagen = Image.open('Imagenes/pandas_9.png')
        st.image(imagen, width=1478) 
        imagen = Image.open('Imagenes/pandas_10.png')
        st.image(imagen, width=1480) 
        st.write('---')

        st.write('##### DataSet')
        st.write('''Un DataSet son los datos que estan organizados de cierta manera en un archivo txt, csv, xlsx, etc.
    ''')
        
        st.write('''
        Parámetros del read_csv 
        - sep: el caracter utilizado para separar los valores (delimitador). El predeterminado es la coma ','.
        - header: la fila que se usará como encabezado, header=0 (primera fila) o header=None.
        - names: una lista de nombres de columna para usar en caso de que el archivo no tenga encabezado.
        - index_col: especifica la columna a usar como índice del DataFrame.
        - na_values: se utiliza para especificar que valores deben interpretarse como valores faltantes (NaN) al cargarlo en un DataFrame. 
        Se pueden pasar una lista de cadenas (n/a, ---, ?, etc.) ademas de los valores predeterminados como '', 'NULL', 'NA', etc.
        ''')
        
        imagen = Image.open('Imagenes/pandas_11.png')
        st.image(imagen, width=1478) 
        imagen = Image.open('Imagenes/pandas_12.png')
        st.image(imagen, width=1477) 
        imagen = Image.open('Imagenes/pandas_13.png')
        st.image(imagen, width=1478) 
        imagen = Image.open('Imagenes/pandas_14.png')
        st.image(imagen, width=1480) 
        st.write('---')

        st.write('##### Acceder a columnas y filas')     
        imagen = Image.open('Imagenes/pandas_18.png')
        st.image(imagen, width=1478)                       
 
        st.write('''
        ##### Acceder por el metodo loc:
        Se utiliza para seleccionar datos de un DataFrame utilizando etiquetas (nombres) de fila y columna  
        ''')
     
        imagen = Image.open('Imagenes/pandas_19.png')
        st.image(imagen, width=1476)              
        imagen = Image.open('Imagenes/pandas_20.png')
        st.image(imagen, width=1478)          
        
        st.write('''
        ##### Acceder por el metodo iloc:
        Permite seleccionar los datos en un DataFrame utilizando posiciones enteras  
        ''')
        imagen = Image.open('Imagenes/pandas_21.png')
        st.image(imagen, width=1481)         
        st.write('---')

        st.write('##### Transponer un DataFrame')
        imagen = Image.open('Imagenes/pandas_51.png')
        st.image(imagen, width=1481)   


    if opcion_seleccionada == 'Exploración de Datos':         
        imagen = Image.open('Imagenes/pandas_15.png')
        st.image(imagen, width=1479) 
        imagen = Image.open('Imagenes/pandas_16.png')
        st.image(imagen, width=1478)     
        imagen = Image.open('Imagenes/pandas_17.png')
        st.image(imagen, width=1476)      
        imagen = Image.open('Imagenes/pandas_24.png')        
        st.image(imagen, width=1479)         
        st.write('---')

        st.write('##### Renombrar columnas')     
        imagen = Image.open('Imagenes/pandas_22.png')
        st.image(imagen, width=1476)   
        st.write('---')
        
        st.write('##### Valores unicos')     
        imagen = Image.open('Imagenes/pandas_23.png')
        st.image(imagen, width=1478)   
        st.write('---')

        st.write('##### Reemplazar valores')     
        imagen = Image.open('Imagenes/pandas_25.png')
        st.image(imagen, width=1477)   
        
        imagen = Image.open('Imagenes/pandas_39.png')
        st.image(imagen, width=1479)          
        
        imagen = Image.open('Imagenes/pandas_26.png')
        st.image(imagen, width=1477)   
        imagen = Image.open('Imagenes/pandas_28.png')
        st.image(imagen, width=1478)           
        
        st.write('---')
        
        st.write('##### Filtrado de datos')     
        imagen = Image.open('Imagenes/pandas_27.png')
        st.image(imagen, width=1479)   
        st.write('---')        
        
        st.write('##### Matriz de Correlación')     
        imagen = Image.open('Imagenes/pandas_52.png')
        st.image(imagen, width=1479)   
        st.write('---')             
        
        
    if opcion_seleccionada == 'Conversión de Tipos':   
        imagen = Image.open('Imagenes/pandas_29.png')
        st.image(imagen, width=1478)           
        
        st.write('##### Convertir a tipo numerico')     
        imagen = Image.open('Imagenes/pandas_30.png')
        st.image(imagen, width=1482) 

        st.write('##### Convertir a tipo fecha')     
        imagen = Image.open('Imagenes/pandas_31.png')
        st.image(imagen, width=1476) 

        st.write('##### Obtener años/mes/dia de un campo DateTime')   
        imagen = Image.open('Imagenes/pandas_32.png')
        st.image(imagen, width=1478) 
        
        
        st.write('#### Obtener tupla con las columnas de un tipo determinado')
        imagen = Image.open('Imagenes/pandas_55.png')
        st.image(imagen, width=1479)        
        
        st.write('---')
                  

    if opcion_seleccionada == 'Limpieza y Manipulación de Datos':   
        st.write('##### Valores faltantes')     
        imagen = Image.open('Imagenes/pandas_40.png')
        st.image(imagen, width=1479)
        st.write('---')   
    
        st.write('##### Rellenar valores faltantes')    
        imagen = Image.open('Imagenes/pandas_41.png')
        st.image(imagen, width=1479) 
        imagen = Image.open('Imagenes/pandas_42.png')
        st.image(imagen, width=1475) 

        st.write('---')

        st.write('##### Eliminar valores faltantes')  
        imagen = Image.open('Imagenes/pandas_43.png')
        st.image(imagen, width=1481) 

        st.write('---')

        st.write('##### Eliminar columna')  
        imagen = Image.open('Imagenes/pandas_44.png')
        st.image(imagen, width=1481) 

        st.write('---')

        st.write('##### Eliminar fila')  
        imagen = Image.open('Imagenes/pandas_45.png')
        st.image(imagen, width=1479) 

        st.write('##### Eliminar filas duplicadas')  
        imagen = Image.open('Imagenes/pandas_46.png')
        st.image(imagen, width=1479) 

        st.write('---')
        
        st.write('##### Agrupación')  
        imagen = Image.open('Imagenes/pandas_47.png')
        st.image(imagen, width=1479)         
        imagen = Image.open('Imagenes/pandas_48.png')
        st.image(imagen, width=1480)  
        
        
        st.write('---')
        
        st.write('##### Ordenar')          
        imagen = Image.open('Imagenes/pandas_49.png')
        st.image(imagen, width=1479)          
        imagen = Image.open('Imagenes/pandas_50.png')
        st.image(imagen, width=1479)                  

    if opcion_seleccionada == 'Fusionar, Combinar y Concatenar Data Frames': 
        st.write('''
            **merge():** fusiona dos DataFarmes basandose en valores comunes de una o mas columnas.  
        ''') 
        imagen = Image.open('Imagenes/pandas_34.png')
        st.image(imagen, width=1476) 
        imagen = Image.open('Imagenes/pandas_35.png')
        st.image(imagen, width=1474) 
        st.write('---')
        
        st.write('''
            **join():**: permite unir dos DataFrames a partir de un indice o una columna clave .  
        ''') 
        imagen = Image.open('Imagenes/pandas_36.png')
        st.image(imagen, width=1475) 
        st.write('---')
        
        st.write('''
            **concat():**: permite unir dos DataFrames a partir de un eje (vertical o horizontal) .  
        ''')         
        imagen = Image.open('Imagenes/pandas_37.png')
        st.image(imagen, width=1477)       
        imagen = Image.open('Imagenes/pandas_38.png')
        st.image(imagen, width=1480)            

    if opcion_seleccionada == 'Respaldos':    
        imagen = Image.open('Imagenes/pandas_53.png')
        st.image(imagen, width=1481)
        st.write('---')   

    if opcion_seleccionada == 'Reporte ydata-profiling':    
        imagen = Image.open('Imagenes/pandas_54.png')
        st.image(imagen, width=1482)


def matplotlib():

    opciones_plt = ['plot','scatter','hist','bar','boxplot','pie']
    
    # Carga de Datos
    df = pd.read_csv('DataFrames/tips.csv')
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_plt)
        st.success(f'##### **{opcion_seleccionada}** ')
    
    if opcion_seleccionada == 'plot':
        st.write('Función que genera gráficos de línea')
        
        codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
        st.code(codigo)
        st.dataframe(df.head())


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
        ''')

        codigo = '''fig,ax = plt.subplots()   
ax.scatter(x=df['total_bill'], y=df['tip'], color='#ff8c00')
ax.set_title('Diagrama de Dispersión (Total de la Cuenta vs Propina)')
ax.set_xlabel('Total de la cuenta')
ax.set_ylabel('Propina')

st.pyplot(fig)'''     
        st.code(codigo)

        # scatter
        with st.container(width=800):
            fig,ax = plt.subplots()

            ax.scatter(x=df['total_bill'], y=df['tip'], color='#ff8c00')
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
        * bins -> define el nro de columnas que se muestran en el grafico         
        * edgecolor -> estable el color de los bordes de las barras.  
        * color -> define el color de la barra   
        * alpha -> determina la opacidad''')


        codigo = '''fig,ax = plt.subplots()   
ax.hist(x=df['total_bill'], bins=15, edgecolor='#000000', color='#8b92cc', alpha=.8)
ax.set_title('Histograma (Distribución del Total de la Cuenta)')
ax.set_xlabel('Total de la Cuenta')
ax.set_ylabel('Frecuencia')

st.pyplot(fig)'''     
        st.code(codigo)

        # hist
        with st.container(width=800):
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
            
            
        codigo = '''fig,ax = plt.subplots()   
ax.bar(df['day'],df['total_bill'])
ax.set_title('Total de la Cuenta por día')
ax.set_xlabel('Día')
ax.set_ylabel('Total de la Cuenta')

st.pyplot(fig)'''     
        st.code(codigo)

        
        with st.container(width=800):
            fig,ax = plt.subplots()

            ax.bar(df['day'],df['total_bill'])
            ax.set_title('Total de la Cuenta por día')
            ax.set_xlabel('Día')
            ax.set_ylabel('Total de la Cuenta')
            st.pyplot(fig)            
            
            
 
    if opcion_seleccionada == 'boxplot':
        st.write('Función para crear diagramas de caja.')
        
        codigo = '''import matplotlib.pyplot as plt
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()'''
        st.code(codigo)
        st.dataframe(df.head())              
            
 
 
        codigo = '''fig,ax = plt.subplots()   
ax.boxplot(df['total_bill'])
ax.set_title('Total de la Cuenta por día')
ax.set_ylabel('Total de la Cuenta')

st.pyplot(fig)'''     
        st.code(codigo)

        
        with st.container(width=800):
            fig,ax = plt.subplots()

            ax.boxplot(df['total_bill'])
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
        * autopct -> de que forma se ve el valor del %          
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
    
    opciones_sns = ['displot','jointplot','pairplot','rugplot','kdeplot','barplot']
    
    # Carga de Datos
    df = pd.read_csv('DataFrames/tips.csv')
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_sns)
        st.success(f'##### **{opcion_seleccionada}** ')
    
    if opcion_seleccionada == 'displot':
        st.write('Gráfico que se utiliza para conjuntos de observaciones univariantes y los visualiza mediante un histograma.')
        
        st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
        st.dataframe(df.head())

        st.write('##### Parámetros')
        
        st.write('''
        * bins -> define los intervalos en un histograma.        
        * kde -> crea una curva suave que representa la función de densidad de probabilidad de los datos.     
''')
        
        st.code('''graf = sns.displot(
    data = df,
    x='total_bill',
    bins=20,
    kde=True
)

st.pyplot(graf)''')

        # displot
        with st.container(border=True, width=800):
            graf = sns.displot(
                data = df,
                x='total_bill',
                bins=20, 
                kde=True
            )
            
            st.pyplot(graf)  
 
    if opcion_seleccionada == 'jointplot':
        st.write('''Gráfico que se utiliza para visualizar la relación entre dos variables.    
Combina un gráfico bivariado en el centro con histogramas o gráficos de densidad en los márgenes para mostrar la distribución''')
        
        st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
        st.dataframe(df.head())

        st.write('##### Parámetros')
        
        st.write('''
        * kind -> permite definir el tipo de representacion (hex=hexagonal)(reg=regresion)(kde=densidad de puntos)        
''')
        
        st.code('''graf = sns.jointplot(
    data = df,
    x='total_bill',
    y='tip',
    kind='reg'
)

st.pyplot(graf)''')

        # displot
        with st.container(border=True, width=800):
            graf = sns.jointplot(
                data = df,
                x='total_bill',
                y='tip',
                kind='reg'
            )
            
            st.pyplot(graf)  
 
 
    if opcion_seleccionada == 'pairplot':
        st.write('''Crea una matriz de visulaizaciones para explorar relaciones entre variables en un conjunto de datos.    
La función grafica diagramas de dispersión para las relaciones por pares y utiliza gráficos univariados en la diagonal para mostrar la distribución de cada variable individual.''')
        
        st.code('''import seaborn as sns
tips = pd.read_csv('Archivos/tips.csv')     # Carga de DataFrame
tips.head()''')
        st.dataframe(df.head())

        st.write('##### Parámetros')
        
        st.write('''
        * hue -> argumento para diferenciar colores por categoria.    
        * palette -> define la paleta de colores: deep, muted, bright, pastel, dark, colorblind, husl, RdYBlyu, magma, YlOrBr, crest, rocket_r, mako, viridis''')
        
        st.code('''graf = sns.pairplot(
    data = df,
    hue= 'sex',
    palette='rocket_r'
)

st.pyplot(graf)''')

        # displot
        with st.container(border=True):
            graf = sns.pairplot(
                data = df,
                hue= 'sex',
                palette='rocket_r'
            )
            
            st.pyplot(graf)   
 
 
 
 
 
 
 
 
 
            
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


if __name__ == '__main__':
    main()

