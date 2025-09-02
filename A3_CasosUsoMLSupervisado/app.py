from flask import Flask 
from flask import render_template
app = Flask(__name__)

@app.route("/")
def home ():
     myname = None
     myname = "Flask" 
     return f"Hello, {myname}!"

@app.route('/index')
def index():
     myname = "Flask"
     return render_template('index.html',name=myname)

if __name__ == '__main__':
     app.run(debug=True)