import streamlit as st
from PIL import Image

def main():
    # Crear sidebar con opciones
    #st.sidebar.header('Seleccionar Opcion:')
    
    # Lista de opciones del sidebar
    opciones = ['Seleccionar...', 'Python','Numpy','Pandas','Matplotlib','Seaborn','Plotly']
    
    opcion_seleccionada = st.sidebar.selectbox(
        '**Bibliotecas:**',
        opciones
    )
    
    st.sidebar.write('---')
    
    # Lista de opciones del sidebar
    opciones_2 = ['Seleccionar...', 'Análisis Exploratorio de Datos (EDA)', 'Modelado de Datos (ML)']
    
    opcion_seleccionada_2 = st.sidebar.selectbox(
        '**Data Science - Machine Learning:**',
        opciones_2
    )
    
    st.sidebar.write('---')    
    
    opciones_3 = ['Seleccionar...','Git y GitHub','AWS','Docker','Streamlit']
 
    opcion_seleccionada_3 = st.sidebar.selectbox(
        '**Herramientas:**',
        opciones_3
    )
    
    
    st.sidebar.info('Ultima Actualización: 09/11/2025')
    
    if opcion_seleccionada != 'Seleccionar...':
        st.title(opcion_seleccionada)
    
        if opcion_seleccionada == 'Python':
            python()
        elif opcion_seleccionada == 'Numpy':
            numpy()
        elif opcion_seleccionada == 'Pandas':
            pandas()
            
    if opcion_seleccionada_3 != 'Seleccionar...':
        st.title(opcion_seleccionada_3)
        
        if opcion_seleccionada_3 == 'Git y GitHub':
            git()
        
        
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
        st.markdown('''
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
        st.markdown('''
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
        st.markdown('''
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
        st.markdown('''Verifica que los elementos de una secuencia cumplan una condicion, 
        devolviendo un iterador con los elementos que cumplen dicha condicion.
    ''')

        imagen = Image.open('Imagenes/filter.png')
        st.image(imagen, width=1479)
 
        st.write('---')
        st.write('##### Referencias externas')    
             
    if opcion_seleccionada == 'map':         
        st.markdown('''Aplica una funcion a cada elemento de un iterable devolviendo una lista con los resultados.
    ''')

        imagen = Image.open('Imagenes/map.png')
        st.image(imagen, width=1478)
 
        st.write('---')
        st.write('##### Referencias externas')            
 
    if opcion_seleccionada == 'reduce':         
        st.markdown('''Aplica una funcion que calcula el produco de todos los elementos de una lista.
    ''')

        imagen = Image.open('Imagenes/reduce.png')
        st.image(imagen, width=1480)
 
        st.write('---')
        st.write('##### Referencias externas')    
 
    if opcion_seleccionada == 'Generadores':         
        st.markdown('''Es una funcion que devuelve varios valores en tiempo de ejecucion.
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
        st.markdown('''Es una funcion que crear un array (vector o matriz).
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
        st.markdown('''Funcion que crea numeros pseudoaleatorios.
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
        st.markdown('''Una Serie es una estructura de datos unidimensional que puede contener cualquier tipo de datos.
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
        st.markdown('''Un DataFrame es una estructura de datos bidimensional con etiquetas que se asemeja a una hoja de cálculo o una tabla
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
        st.markdown('''Un DataSet son los datos que estan organizados de cierta manera en un archivo txt, csv, xlsx, etc.
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
 
        st.markdown('''
        ##### Acceder por el metodo loc:
        Se utiliza para seleccionar datos de un DataFrame utilizando etiquetas (nombres) de fila y columna  
        ''')
     
        imagen = Image.open('Imagenes/pandas_19.png')
        st.image(imagen, width=1476)              
        imagen = Image.open('Imagenes/pandas_20.png')
        st.image(imagen, width=1478)          
        
        st.markdown('''
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
        st.markdown('''
            **merge():** fusiona dos DataFarmes basandose en valores comunes de una o mas columnas.  
        ''') 
        imagen = Image.open('Imagenes/pandas_34.png')
        st.image(imagen, width=1476) 
        imagen = Image.open('Imagenes/pandas_35.png')
        st.image(imagen, width=1474) 
        st.write('---')
        
        st.markdown('''
            **join():**: permite unir dos DataFrames a partir de un indice o una columna clave .  
        ''') 
        imagen = Image.open('Imagenes/pandas_36.png')
        st.image(imagen, width=1475) 
        st.write('---')
        
        st.markdown('''
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


def git():

    opciones_git = ['Git', 'GitHub']
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_git)
        st.success(f'##### **{opcion_seleccionada}** ')
    
    # ------------------------------------------- PRINT() -------------------------------------------------------
    if opcion_seleccionada == 'Git':
        st.markdown('''**Definción:** es un sistema de Control de Versiones.    
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

