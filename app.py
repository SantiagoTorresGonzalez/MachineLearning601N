from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import io
import base64
import os
import matplotlib.pyplot as plt

import prediccionTrafico
import randomForest
from regresionAccidente import logistic_Model, scaler, columnas_modelo, predict_label

app = Flask(__name__)

# ---------------- RUTAS PRINCIPALES ----------------
@app.route('/')
def index():
    return render_template('index.html', name="Flask")

@app.route('/PrimerCaso')
def PrimerC():
    return render_template('PrimerCaso.html', name="Flask")

@app.route('/SegundoCaso')
def SegundoC():
    return render_template('SegundoC.html', name="Flask")

@app.route('/TercerCaso')
def TercerC():
    return render_template('TercerCaso.html', name="Flask")

@app.route('/CuartoCaso')
def CuartoC():
    return render_template('CuartoCaso.html', name="Flask")

@app.route('/ConceptosBasicos')
def ConcepB():
    return render_template('ConceptosBasicos.html', name="Flask")

@app.route('/ConceptosBasicos2')
def ConceptosBasicos2():
    return render_template('ConceptosBasicos2.html', name="Flask")

@app.route('/ConceptosBasicosRandomForest')
def ConceptosBasicosRandomForest():
    return render_template('ConceptosBasicosRandomForest.html', name="Flask")

# ---------------- PREDICCIÓN TRÁFICO ----------------
@app.route('/EjercicioPractico', methods=['GET', 'POST'])
def EjercicioPractico():
    prediction = None
    distancia = request.form.get('distancia', '')
    trafico = request.form.get('trafico', '')

    if request.method == 'POST':
        try:
            d = float(distancia)
            t = float(trafico)
            prediction = round(prediccionTrafico.CalcularTiempoProm(d, t), 2)
        except Exception as e:
            prediction = f"Error en los valores: {e}"

    return render_template('EjercicioPractico.html', prediction=prediction, distancia=distancia, trafico=trafico)

@app.route('/plot.png')
def plot_png():
    fig = prediccionTrafico.crear_fig()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# ---------------- REGRESIÓN LOGÍSTICA ----------------
@app.route("/EjercicioPractico2")
def EjercicioPractico2():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, "static", "accuracy.txt"), "r") as f:
        accuracy = f.read().strip()
    return render_template("EjercicioPractico2.html", accuracy=accuracy)

@app.route("/predecir", methods=["POST"])
def predecir():
    velocidad = float(request.form["velocidad"])
    edad = float(request.form["edad"])
    clima = request.form["clima"]
    estado_via = request.form["estado_via"]

    input_data = pd.DataFrame({
        "Velocidad": [velocidad],
        "EdadConductor": [edad],
        "Clima": [clima],
        "EstadoVia": [estado_via]
    })

    input_data = pd.get_dummies(input_data, columns=["Clima", "EstadoVia"])
    input_data = input_data.reindex(columns=columnas_modelo, fill_value=0)

    label, prob = predict_label(logistic_Model, scaler, input_data)

    return render_template(
        "EjercicioPractico2.html",
        prediccion=label,
        probabilidad=f"{prob*100:.2f}"
    )

# ---------------- CLASIFICACION - DIABETES ----------------
@app.route("/diabetes", methods=["GET", "POST"])
def Diabetes():
    resultado = None
    probabilidad = None
    interpretacion = None

    if request.method == "POST":
        edad = float(request.form["edad"])
        imc = float(request.form["imc"])
        glucosa = float(request.form["glucosa"])
        presion = float(request.form["presion"])
        historial = request.form["historial"]

        try:
            historial_num = randomForest.le_fam.transform([historial.strip()])[0]
        except ValueError:
            return render_template("diabetes.html",
                                   prediction="Error: valor de historial no reconocido")

        # Features formateados para el modelo
        features = np.array([[edad, imc, glucosa, presion, historial_num]])
        
        # Probabilidad de la clase positiva
        proba = randomForest.bosque.predict_proba(features)[0]
        prob_diabetes = round(proba[1], 4)  # 1 corresponde a "Diabetes"

        threshold = 0.4
        pred = 1 if prob_diabetes >= threshold else 0
        resultado = randomForest.le_diag.inverse_transform([pred])[0]

        probabilidad = prob_diabetes
        interpretacion = "Con threshold=0.4, el modelo se vuelve más sensible (detecta más diabéticos), Pero a cambio pierde precisión en los sanos (personas sin diabetes)."

    # leer exactitud desde randomForest (ya precomputada)
    return render_template(
        "diabetes.html",
        prediction=resultado,
        probabilidad=probabilidad,
        interpretacion=interpretacion,
        accuracy=randomForest.accuracy_val
    )


@app.route("/plot_tree")
def plot_tree():
    fig = randomForest.plot_tree_example()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

# ---------------- EJECUCIÓN ----------------
if __name__ == '__main__':
    app.run(debug=True)
