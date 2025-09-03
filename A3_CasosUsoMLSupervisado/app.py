from flask import Flask 
from flask import render_template
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

if __name__ == '__main__':
     app.run(debug=True)