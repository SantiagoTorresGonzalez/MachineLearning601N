from flask import Flask, render_template, request, send_file
import io
import matplotlib.pyplot as plt
import prediccionTrafico  # tu script.py con modelo, CalcularTiempoProm(...) y crear_fig()

app = Flask(__name__)

# Rutas que ya tenías
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


# Ruta que devuelve la imagen PNG generada por script.crear_fig()
@app.route('/plot.png')
def plot_png():
    fig = prediccionTrafico.crear_fig()  # crear_fig() debe estar en script.py
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
