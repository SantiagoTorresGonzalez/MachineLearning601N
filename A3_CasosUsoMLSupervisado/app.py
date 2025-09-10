import regresionLineal
import pandas as pd
import numpy as np
from flask import Flask, request
from flask import render_template
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import predicciónTrafico
import io, base64
app= Flask(__name__)

datos = {
    'Distancia_km': [5, 10, 3, 8, 12, 28, 27, 4, 15],
    'Trafico_promedio': [200, 350, 100, 250, 400, 370, 500, 380, 200],
    'Tiempo_entrega': [15, 30, 8, 22, 40, 50, 30, 61, 7]
}

df = pd.DataFrame(datos)
x = df[['Distancia_km', 'Trafico_promedio']]
y = df[['Tiempo_entrega']]

# -----------------------------
# 2. Entrenamiento del modelo
# -----------------------------
modelo = LinearRegression()
modelo.fit(x, y)

# -----------------------------
# 3. Función de predicción
# -----------------------------
def CalcularTiempoProm(distancia, trafico):
    result = modelo.predict([[distancia, trafico]])[0][0]
    return round(result, 2)

app = Flask(__name__)

@app.route('/')
def index():
     myname = "Flask"
     return render_template('index.html',name=myname)

@app.route('/PrimerCaso')
def PrimerC():
     myname = "Flask"
     return render_template('PrimerCaso.html',name=myname)

@app.route('/SegundoCaso')
def SegundoC():
     myname = "Flask"
     return render_template('SegundoC.html',name=myname)

@app.route('/TercerCaso')
def TercerC():
     myname = "Flask"
     return render_template('TercerCaso.html',name=myname)

@app.route('/CuartoCaso')
def CuartoC():
     myname = "Flask"
     return render_template('CuartoCaso.html',name=myname)

@app.route('/ConceptosBasicos')
def ConcepB():
     myname = "Flask"
     return render_template('ConceptosBasicos.html',name=myname)

@app.route('/EjercicioPractico')
def EjerP():
     myname = "Flask"
     # Si el usuario envía datos en el formulario
    if request.method == "POST":
        distancia = int(request.form["distancia"])
        trafico = int(request.form["trafico"])
        prediccion = predicciónTrafico.CalcularTiempoProm(distancia, trafico)

    # Dataset para mostrar en tabla y gráfico
    df = pd.DataFrame(predicciónTrafico.datos)

    # Crear gráfico
    plt.figure(figsize=(6, 4))
    plt.scatter(df["Distancia_km"], df["Tiempo_entrega"], color="blue", label="Datos reales")

    # Ordenar las distancias para la línea
    distancias_ordenadas = np.linspace(df["Distancia_km"].min(), df["Distancia_km"].max(), 100)

    # Usar el tráfico promedio como referencia
    trafico_promedio = df["Trafico_promedio"].mean()

    # Generar predicciones con el modelo
    y_pred_linea = predicciónTrafico.modelo.predict(
        np.column_stack([distancias_ordenadas, np.full_like(distancias_ordenadas, trafico_promedio)])
    )

    # Dibujar línea de regresión
    plt.plot(distancias_ordenadas, y_pred_linea, color="red", label="Línea de regresión")

    plt.xlabel("Distancia (km)")
    plt.ylabel("Tiempo de entrega (min)")
    plt.title("Relación entre distancia y tiempo de entrega (con tráfico promedio)")
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    grafico_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return render_template(
    "EjercicioPractico.html",
    tablas=df.to_html(classes="table table-striped", index=False),
    grafico=grafico_base64,
    prediccion=prediccion
    )
     

if __name__ == '__main__':
     app.run(debug=True)
