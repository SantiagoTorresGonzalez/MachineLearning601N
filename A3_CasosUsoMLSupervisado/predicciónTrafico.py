import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Se crea el diccionario de datos (distancia, tráfico y tiempo de entrega)
datos= {
    'Distancia_km': [5, 10, 3, 8, 12, 28, 27, 4, 15],
    'Trafico_promedio': [200, 350, 100, 250, 400, 370, 500, 380, 200],
    'Tiempo_entrega': [15, 30, 8, 22, 40, 50, 30, 61, 7]
}

#Se asignan las variables y que valores abarcan
df= pd.DataFrame(datos)
x= df[['Distancia_km', 'Trafico_promedio']]
y= df[['Tiempo_entrega']]

#Se llama la regresión lineal
modelo= LinearRegression()
#Se adapta la regresión a las variables requeridas en el ejercicio
modelo.fit(x, y)


#Se realiza la predicción
def CalcularTiempoProm(distancia, trafico):
    result= modelo.predict([[distancia, trafico]])[0][0]
    return result 
