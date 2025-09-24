from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import prediccionTrafico
import randomForest
from regresionAccidente import logistic_Model, scaler, columnas_modelo, predict_label
import randomForest

app = Flask(__name__)

@app.route('/')
def index():
    myname = "Flask"
    return render_template('index.html', name=myname)

@app.route('/PrimerCaso')
def PrimerC():
    myname = "Flask"
    return render_template('PrimerCaso.html', name=myname)

@app.route('/SegundoCaso')
def SegundoC():
    myname = "Flask"
    return render_template('SegundoC.html', name=myname)

@app.route('/TercerCaso')
def TercerC():
    myname = "Flask"
    return render_template('TercerCaso.html', name=myname)

@app.route('/CuartoCaso')
def CuartoC():
    myname = "Flask"
    return render_template('CuartoCaso.html', name=myname)

@app.route('/ConceptosBasicos')
def ConcepB():
    myname = "Flask"
    return render_template('ConceptosBasicos.html', name=myname)

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

# Esta ruta devuelve la imagen de la matriz en .png
@app.route('/plot.png')
def plot_png():
    fig = prediccionTrafico.crear_fig() 
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/ConceptosBasicos2')
def ConceptosBasicos2():
    myname = "Flask"
    return render_template('ConceptosBasicos2.html', name=myname)

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

@app.route('/ConceptosBasicosRandomForest')
def ConceptosBasicosRandomForest():
    myname = "Flask"
    return render_template('ConceptosBasicosRandomForest.html', name=myname)

@app.route("/diabetes", methods=["GET", "POST"])
def Diabetes():
    resultado = None
    if request.method == "POST":
        edad = float(request.form["edad"])
        imc = float(request.form["imc"])
        glucosa = float(request.form["glucosa"])
        presion = float(request.form["presion"])
        historial = request.form["historial"]  # "Sí" o "No"

        # Usar directamente el modelo y encoders de randomForest.py
        # Esto asume que randomForest.py tiene: bosque, le_fam, le_diag
        historial_num = randomForest.le_fam.transform([historial.strip()])[0]

        features = np.array([[edad, imc, glucosa, presion, historial_num]])
        pred = randomForest.bosque.predict(features)[0]

        resultado = randomForest.le_diag.inverse_transform([pred])[0]

    return render_template("diabetes.html", prediction=resultado)


if __name__ == '__main__':
    app.run(debug=True)
