import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import Proyecciones_reales as pr
import Proyecciones as pn

app = Flask(__name__)
app.config['UPLOAD_FOLDER']="./Datos"

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/Proyecciones_reales')
def p_reales():
    return render_template('p_reales.html')


@app.route('/Proyecciones_sinteticas')
def p_sinteticas():
    return render_template('p_sinteticas.html')

@app.route('/Subir_Archivo_CSV')
def cargar():
    return render_template('cargar.html')

@app.route("/upload", methods=['POST'])
def uploader():
  if request.method == 'POST':
     f = request.files['archivo_csv']
     filename = secure_filename(f.filename)
     f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
     return render_template('test.html')

@app.route("/test")
def test():
     return render_template('test.html')

@app.route('/reprobacion')
def reprobacion():
   return render_template('P_Reprobacion.html')
   

@app.route('/desercion')
def desercion():
   return render_template('P_Desercion.html')

@app.route('/repitencia')
def repitencia():
   return render_template('P_Repitencia.html')

@app.route("/run2", methods=['GET'])
def run_pro_reprobacion():
    if request.method == 'GET':
        coe,mse,ve = pn.proyeccion_reprobacion()
        return render_template('P_Reprobacion.html',ceo=coe,mse=mse,ve=ve)

@app.route("/run3", methods=['GET'])
def run_pro_desercion():
    if request.method == 'GET':
        pn.proyeccion_desercion()
        return render_template('p_reales.html')

@app.route("/run4", methods=['GET'])
def run_pro_repitencia():
    if request.method == 'GET':
        pn.proyeccion_repitencia()
        return render_template('p_reales.html')


if __name__ == "__main__":
    app.run(debug=True)
