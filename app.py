from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import io
import base64
import os
import matplotlib.pyplot as plt

import prediccionTrafico
import randomForest
from regresionAccidente import logistic_Model, scaler, columnas_modelo, predict_label
from gridworld import GridWorld
from agent import QLearningAgent
from training import entrenar

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', name="Flask")

@app.route('/PrimerCaso')
def PrimerC():
    return render_template('PrimerCaso.html', name="Flask")

@app.route('/SegundoCaso')
def SegundoC():
    return render_template('SegundoCaso.html', name="Flask")

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
            return render_template("diabetes.html", prediction="Error: valor de historial no reconocido")

        features = np.array([[edad, imc, glucosa, presion, historial_num]])

        label, prob = randomForest.predict_label(features, threshold=0.5)
        resultado = label
        probabilidad = f"{prob:.4f}"    
        interpretacion = (
            "Con threshold=0.4, el modelo se vuelve más sensible (detecta más diabéticos), "
            "pero pierde precisión en los sanos."
        )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, "static", "accuracy_diabetes.txt"), "r") as f:
        accuracy = f.read().strip()

    return render_template(
        "diabetes.html",
        prediction=resultado,
        probabilidad=probabilidad,
        interpretacion=interpretacion,
        accuracy=accuracy
    )

# APRENDIZAJE POR REFUERZO – GRIDWORLD

@app.route("/gridworld/teoria")
def gridworld_teoria():
    return render_template("gridworld_teoria.html")

@app.route("/gridworld/practica")
def gridworld_practica():
    return render_template("gridworld_practica.html")

# Inicializar entorno y agente ====
env = GridWorld()
agent = QLearningAgent(env.get_actions())
IMG_PATH = "static/trayectoria.png"

# ENTRENAR ====
@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json()
    episodios = int(data["episodios"])

    historial = entrenar(env, agent, episodios)
    recompensa_prom = sum(historial) / len(historial)

    return jsonify({
        "status": "ok",
        "episodios": episodios,
        "epsilon": round(agent.epsilon, 3),
        "recompensa_promedio": round(recompensa_prom, 3),
        "historial": historial
    })

# PROBAR AGENTE====
@app.route("/api/test")
def api_test():
    ruta = []
    estado = env.reset()
    ruta.append(estado)

    done = False
    while not done:
        accion = agent.choose_action(estado)
        nuevo_estado, _, done = env.step(accion)
        estado = nuevo_estado
        ruta.append(estado)

    _dibujar_ruta(ruta)
    return jsonify({"status": "ok", "ruta": ruta})

# SERVIR IMAGEN ====
@app.route("/api/trayectoria")
def api_trayectoria():
    if os.path.exists(IMG_PATH):
        return send_file(IMG_PATH, mimetype="image/png")
    return jsonify({"error": "Imagen no generada"}), 404

# FUNCIÓN PARA DIBUJAR ====
def _dibujar_ruta(ruta):
    filas, cols = env.mapa.shape

    fig, ax = plt.subplots()
    ax.imshow(env.mapa, cmap="coolwarm", alpha=0.6)

    xs = [p[1] for p in ruta]
    ys = [p[0] for p in ruta]
    ax.plot(xs, ys, marker="o")
    ax.set_title("Trayectoria del agente")

    fig.savefig(IMG_PATH)
    plt.close(fig)

# REINICIAR EL AGENTE
@app.route("/api/reset")
def api_reset():
    global agent, env

    env = GridWorld()                       
    agent = QLearningAgent(env.get_actions())  

    return jsonify({"status": "reset_ok"})

if __name__ == '__main__':
    app.run(debug=True)