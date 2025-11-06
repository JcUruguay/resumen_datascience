import streamlit as st
from PIL import Image

def main():
    # Crear sidebar con opciones
    #st.sidebar.header('Seleccionar Opcion:')
    
    # Lista de opciones del sidebar
    opciones = ['Seleccionar...', 'Python','Numpy','Pandas','Matplotlib','Seaborn','Plotly']
    
    opcion_seleccionada = st.sidebar.selectbox(
        'Bibliotecas: ',
        opciones
    )
    
    if opcion_seleccionada != 'Seleccionar...':
        st.title(opcion_seleccionada)
    
        if opcion_seleccionada == 'Python':
            python()
        elif opcion_seleccionada == 'Numpy':
            numpy()
        elif opcion_seleccionada == 'Pandas':
            pandas()
        

    
    
def python():

    opciones_py = ['print','format - f-string','type','Conversión de Tipo','Operadores']
    
    opcion_seleccionada = st.selectbox('Seleccionar: ', opciones_py)
    #st.write('---')
    st.write(f'#### **{opcion_seleccionada}** ')
    
    # ------------------------------------------- PRINT() -------------------------------------------------------
    if opcion_seleccionada == 'print':
        st.write('Función que se utiliza para mostrar mensajes en pantalla.')
        imagen = Image.open('Imagenes/Python_1_1.png')
        st.image(imagen, width=1466)
        
        imagen = Image.open('Imagenes/Python_1_2.png')
        st.image(imagen, width=1463)     
        
        imagen = Image.open('Imagenes/Python_1_3.png')
        st.image(imagen, width=1465)         
        
        st.write('---')
        st.write('##### Referencia externas')
        url = 'https://www.w3schools.com/python/ref_func_print.asp'
        st.page_link(url, label='W3 Schools: Funcion print() de Python')
        
    if opcion_seleccionada == 'format - f-string':   
        st.markdown('''
            **format()** es una funcion que permite formatear una cadena de texto.  
            **f-string** permite concatenar diferentes tipos de datos dentro de un string
        ''') 
        imagen = Image.open('Imagenes/Python_2.png')
        st.image(imagen, width=1470)   
        
        st.write('---')
        st.write('##### Referencia externas')
        url = 'https://www.w3schools.com/python/ref_string_format.asp'
        st.page_link(url, label='W3 Schools: Funcion String format() de Python')        
        
        
    if opcion_seleccionada == 'type':      
        st.write('Funcion que devuelve el tipo de dato')
        imagen = Image.open('Imagenes/Python_3.png')
        st.image(imagen, width=1467)
 
        st.write('---')
        st.write('##### Referencia externas')
        url = 'https://www.w3schools.com/python/ref_func_type.asp'
        st.page_link(url, label='W3 Schools: Funcion type() en Python')     
 

    if opcion_seleccionada == 'Conversión de Tipo':   
        st.markdown('''
            **str()** convierte un numero a un string.     
            **float()** convierte un entero o string a un float.        
            **int()** convierte un float o string a un int.
        ''')    
        st.write('Funciones que permiten convertir el tipo de dato')
        imagen = Image.open('Imagenes/Python_4.png')
        st.image(imagen, width=1465)
        
        st.write('---')
        st.write('##### Referencia externas')
        url = 'https://www.w3schools.com/python/ref_func_str.asp'
        st.page_link(url, label='W3 Schools: Funcion str() en Python')     
        url = 'https://www.w3schools.com/python/ref_func_int.asp'
        st.page_link(url, label='W3 Schools: Funcion int() en Python')  
        url = 'https://www.w3schools.com/python/ref_func_float.asp'
        st.page_link(url, label='W3 Schools: Funcion float() en Python')         
        
    if opcion_seleccionada == 'Operadores':      
        st.write('**Operadores Aritmeticos:** Suma, Resta, Multiplicacion, Division, Modulo, Divison entera, Exponente, Raiz cuadrada, Suma de complejos.')
        imagen = Image.open('Imagenes/Python_5.png')
        st.image(imagen, width=1465)
 
        imagen = Image.open('Imagenes/Python_5_2.png')
        st.image(imagen, width=1465)
 
        st.write('---')
        st.write('##### Referencia externas')
        url = 'https://ellibrodepython.com/operadores-aritmeticos'
        st.page_link(url, label='El Libreo de Python: Operadores Aritmeticos')           
        
def numpy():
    pass

def pandas():
    pass




if __name__ == '__main__':
    main()

