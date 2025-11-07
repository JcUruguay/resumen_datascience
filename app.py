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
    
    st.sidebar.info('Ultima Actualización: 06/11/2025')
    
    if opcion_seleccionada != 'Seleccionar...':
        st.title(opcion_seleccionada)
    
        if opcion_seleccionada == 'Python':
            python()
        elif opcion_seleccionada == 'Numpy':
            numpy()
        elif opcion_seleccionada == 'Pandas':
            pandas()
        

    
    
def python():

    opciones_py = ['print','format - f-string','input','type','Conversión de Tipo','Operadores', 'Métodos de Cadenas (strings)','round',
                   'Módulo math','Módulo random','Módulo statistics','Listas','Tuplas','Sets','Diccionarios','range','zip','Condicional If',
                   'Condicional IN - NOT IN', 'Ciclo for', 'Ciclo while','Funciones','Lambda','Filter','Map','Reduce','Generadores','Excepciones'
                   ]
    
    col1, col2 = st.columns([2,2])
    
    with col1:
        opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_py)
        st.write(f'#### **{opcion_seleccionada}** ')
    
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
        st.write('##### Referencias externas')
        url = 'https://www.w3schools.com/python/ref_func_print.asp'
        st.page_link(url, label='W3 Schools: Funcion print() de Python')
        
    if opcion_seleccionada == 'format - f-string':   
        st.markdown('''
            **format()** es una funcion que permite formatear una cadena de texto.  
            **f-string** permite concatenar diferentes tipos de datos dentro de un string
        ''') 
        imagen = Image.open('Imagenes/format.png')
        st.image(imagen, width=1470)   
        
        st.write('---')
        st.write('##### Referencias externas')
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
 
        st.write('---')
        st.write('##### Referencias externas')     
  
  
     
        
def numpy():
    pass

def pandas():
    pass




if __name__ == '__main__':
    main()

