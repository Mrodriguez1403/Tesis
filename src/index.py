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

@app.route('/Reg_reprobacion')
def Reg_reprobacion():
    M = pn.cargar_datos_reporbacion()
    return render_template('Reg_Reprobacion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
   

@app.route('/Reg_desercion')
def Reg_desercion():
    M = pn.cargar_datos_desercion()
    return render_template('Reg_Desercion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
   

@app.route('/Reg_repitencia')
def Reg_repitencia():
    M = pn.cargar_datos_repitencia()
    return render_template('Reg_Repitencia.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
   

@app.route("/run2", methods=['GET'])
def run_pro_reprobacion():
    if request.method == 'GET':
        m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve = pn.proyeccion_reprobacion()
        return render_template('P_Reprobacion.html',m_coe=m_coe,m_mse=m_mse,m_ve=m_ve,l_coe=l_coe,l_mse=l_mse,l_ve=l_ve,r_coe=r_coe,r_mse=r_mse,r_ve=r_ve)

@app.route("/run3", methods=['GET'])
def run_pro_desercion():
    if request.method == 'GET':
        m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve = pn.proyeccion_desercion()
        return render_template('P_Desercion.html',m_coe=m_coe,m_mse=m_mse,m_ve=m_ve,l_coe=l_coe,l_mse=l_mse,l_ve=l_ve,r_coe=r_coe,r_mse=r_mse,r_ve=r_ve)

@app.route("/run4", methods=['GET'])
def run_pro_repitencia():
    if request.method == 'GET':
        m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve = pn.proyeccion_repitencia()
        return render_template('P_Repitencia.html',m_coe=m_coe,m_mse=m_mse,m_ve=m_ve,l_coe=l_coe,l_mse=l_mse,l_ve=l_ve,r_coe=r_coe,r_mse=r_mse,r_ve=r_ve)

@app.route("/guardar_reprobacion", methods=['GET'])
def guardar_reprobacion():
    if request.method == 'GET':
        pn.guardar_modelo_reprobacion()
        pn.elimiar_modelos_reprobacion()
        M = pn.cargar_datos_reporbacion()
        return render_template('P_Reprobacion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

@app.route("/guardar_desercion", methods=['GET'])
def guardar_desercion():
    if request.method == 'GET':
        pn.guardar_modelo_desercion()
        pn.elimiar_modelos_desercion()
        M = pn.cargar_datos_desercion()
        return render_template('P_Desercion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

@app.route("/guardar_repitencia", methods=['GET'])
def guardar_repitencia():
    if request.method == 'GET':
        pn.guardar_modelo_repitencia()
        pn.elimiar_modelos_repitencia()
        M = pn.cargar_datos_repitencia()
        return render_template('P_Repitencia.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

@app.route("/Pro_reprobacion")
def Pro_reprobacion():
    lista = pn.cargar_lista_reprobacion()
    return render_template('pro_reprobacion.html',lista = lista)

@app.route("/Mostrar_datos_reprobacion", methods=['GET', 'POST'])
def Mostrar_datos_reprobacion():
    select = request.form.get('lista')
    nombre_modelo = str(select)
    ruta_datos,ruta_grafica = pn.buscar_rutas_reprobacion(nombre_modelo)
    M = pn.cargar_datos_Selecionados_repro(ruta_datos)
    lista = pn.cargar_lista_reprobacion()
    return render_template('pro_reprobacion.html',lista = lista,ruta_grafica=ruta_grafica,m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

@app.route("/Mostrar_datos_desercion", methods=['GET', 'POST'])
def Mostrar_datos_desercion():
    select = request.form.get('lista')
    nombre_modelo = str(select)
    ruta_datos,ruta_grafica = pn.buscar_rutas_desercion(nombre_modelo)
    M = pn.cargar_datos_Selecionados_deser(ruta_datos)
    lista = pn.cargar_lista_desercion()
    return render_template('pro_desercion.html',lista = lista,ruta_grafica=ruta_grafica,m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

@app.route("/Mostrar_datos_repitencia", methods=['GET', 'POST'])
def Mostrar_datos_repitencia():
    select = request.form.get('lista')
    nombre_modelo = str(select)
    ruta_datos,ruta_grafica = pn.buscar_rutas_repitencia(nombre_modelo)
    M = pn.cargar_datos_Selecionados_repit(ruta_datos)
    lista = pn.cargar_lista_repitencia()
    return render_template('pro_repitencia.html',lista = lista,ruta_grafica=ruta_grafica,m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
  


    
@app.route("/Pro_desercion")
def Pro_desercion():
    lista = pn.cargar_lista_desercion()
    return render_template('pro_desercion.html',lista = lista)

@app.route("/Pro_repitencia")
def Pro_repitencia():
    lista = pn.cargar_lista_reprobacion()
    return render_template('pro_repitencia.html',lista = lista)

if __name__ == "__main__":
    app.run(debug=True)
