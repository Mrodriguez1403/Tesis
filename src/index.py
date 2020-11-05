import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import Proyecciones_reales as pr
import Proyecciones as pn
import Proyecciones_sinteticas as ps
import grafica_real as gr
import grafica_sintetica as gs
import reg_pro_aprobacion as rpa

app = Flask(__name__)
app.config['UPLOAD_FOLDER']="./Datos"

# funcion para acceder a la ruta de la informacion del proyecto --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/')
def info():
   return render_template('info.html')

# funcion para acceder a la ruta de las graficas de las tasas de riesgo -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/graficas_aprobacion')
def graficas_aprobacion():
    ruta_reg,ruta_pro = rpa.reg_pro_aprobacion()
    return render_template('aprobacion.html',ruta_reg=ruta_reg,ruta_pro=ruta_pro)

 # funcion para acceder a la ruta de las graficas de la tasa de aprobacion  
@app.route('/graficas_riesgo')
def graficas_riesgo():
    g_real = gr.generar_graficas_reales()
    g_sint = gs.generar_graficas_sinteticas()
    return render_template('index.html',g_real=g_real,g_sintetica=g_sint)
   
# funcion para accder a la ruta de las regresiones reales --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/Proyecciones_reales')
def p_reales():
    return render_template('p_reales.html')

# funcion para acceder a la ruta de las regresiones y proyecciones sinteticas ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/Proyecciones_sinteticas')
def p_sinteticas():
    ruta_reprobacion,ruta_desercion,ruta_repitencia,ruta_pro_repro,ruta_pro_deser,ruta_pro_repit = ps.proyecciones_sinteticas()
    return render_template('p_sinteticas.html',ruta_reprobacion=ruta_reprobacion,ruta_desercion=ruta_desercion,ruta_repitencia=ruta_repitencia,ruta_pro_repro=ruta_pro_repro,ruta_pro_deser=ruta_pro_deser,ruta_pro_repit=ruta_pro_repit)

# funcion para acceder a la ruta de SUBIR la Data MEN al proyecto ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/Subir_Archivo_CSV')
def cargar():
    return render_template('cargar.html')

# funcion para subir la Data MEN al proyecto ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/upload", methods=['POST'])
def uploader():
  if request.method == 'POST':
     f = request.files['archivo_csv']
     filename = "Datos_MEN.csv"
     f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
     g_real = gr.generar_graficas_reales()
     g_sint = gs.generar_graficas_sinteticas()
     return render_template('index.html',g_real=g_real,g_sintetica=g_sint)

# @app.route("/test")
# def test():
#      return render_template('test.html')

# funcion para acceder a la ruta de regresion de reprobacion ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/Reg_reprobacion')
def Reg_reprobacion():
    M = pn.cargar_datos_reporbacion()
    return render_template('Reg_Reprobacion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
   
# funcion para acceder a la ruta de regresion de desercion ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/Reg_desercion')
def Reg_desercion():
    M = pn.cargar_datos_desercion()
    return render_template('Reg_Desercion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
   
# funcion para acceder a la ruta de regresion de repitencia ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/Reg_repitencia')
def Reg_repitencia():
    M = pn.cargar_datos_repitencia()
    return render_template('Reg_Repitencia.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])
   
# funcion para realizar la regresion de reprobacion ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/run2", methods=['GET'])
def run_reg_reprobacion():
    if request.method == 'GET':
        m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve = pn.regresion_reprobacion()
        return render_template('Reg_Reprobacion.html',m_coe=m_coe,m_mse=m_mse,m_ve=m_ve,l_coe=l_coe,l_mse=l_mse,l_ve=l_ve,r_coe=r_coe,r_mse=r_mse,r_ve=r_ve)

# funcion para realizar la regresion de desercion ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/run3", methods=['GET'])
def run_reg_desercion():
    if request.method == 'GET':
        m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve = pn.regresion_desercion()
        return render_template('Reg_Desercion.html',m_coe=m_coe,m_mse=m_mse,m_ve=m_ve,l_coe=l_coe,l_mse=l_mse,l_ve=l_ve,r_coe=r_coe,r_mse=r_mse,r_ve=r_ve)

# funcion para realizar la regresion de repitencia ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/run4", methods=['GET'])
def run_reg_repitencia():
    if request.method == 'GET':
        m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve = pn.regresion_repitencia()
        return render_template('Reg_Repitencia.html',m_coe=m_coe,m_mse=m_mse,m_ve=m_ve,l_coe=l_coe,l_mse=l_mse,l_ve=l_ve,r_coe=r_coe,r_mse=r_mse,r_ve=r_ve)

# funcion para guardar el modelo seleccionado de reprobacion ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/guardar_reprobacion", methods=['GET'])
def guardar_reprobacion():
    if request.method == 'GET':
        pn.guardar_modelo_reprobacion()
        pn.elimiar_modelos_reprobacion()
        M = pn.cargar_datos_reporbacion()
        return render_template('Reg_Reprobacion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

# funcion para guardar el modelo seleccionado de desercion ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/guardar_desercion", methods=['GET'])
def guardar_desercion():
    if request.method == 'GET':
        pn.guardar_modelo_desercion()
        pn.elimiar_modelos_desercion()
        M = pn.cargar_datos_desercion()
        return render_template('Reg_Desercion.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

# funcion para guardar el modelo seleccionado de repitencia ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/guardar_repitencia", methods=['GET'])
def guardar_repitencia():
    if request.method == 'GET':
        pn.guardar_modelo_repitencia()
        pn.elimiar_modelos_repitencia()
        M = pn.cargar_datos_repitencia()
        return render_template('Reg_Repitencia.html',m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

#funcion para acceder a la ruta de proyeccion de reprobacion -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/Pro_reprobacion")
def Pro_reprobacion():
    lista = pn.cargar_lista_reprobacion()
    return render_template('pro_reprobacion.html',lista = lista)

#funcion para acceder a la ruta de proyeccion de desercion -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/Pro_desercion")
def Pro_desercion():
    lista = pn.cargar_lista_desercion()
    return render_template('pro_desercion.html',lista = lista)

#funcion para acceder a la ruta de proyeccion de repitencia -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/Pro_repitencia")
def Pro_repitencia():
    lista = pn.cargar_lista_reprobacion()
    return render_template('pro_repitencia.html',lista = lista)
#funcion para acceder a la ruta de proyecciones generales -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/Pro_General")
def Pro_General():
    ruta_repro="static/file/proyecciones/proyeccion_reprobacion.png"
    ruta_deser="static/file/proyecciones/proyeccion_desercion.png"
    ruta_repit="static/file/proyecciones/proyeccion_repitencia.png"
    return render_template('pro_general.html',ruta_repro = ruta_repro,ruta_deser=ruta_deser,ruta_repit=ruta_repit)

#funcion para realizar la proyeccion de reprobacion del modelo seleccionado -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/run_pro_repro", methods=['GET', 'POST'])
def run_pro_repor():
    select = request.form.get('lista_regresion')
    nombre_modelo = str(select)
    ruta_datos,ruta_grafica = pn.buscar_rutas_reprobacion(nombre_modelo)
    M = pn.cargar_datos_Selecionados_repro(ruta_datos)
    ruta_grafica2 = pn.Proyeccion_modelo_reprobacion(nombre_modelo)
    lista = pn.cargar_lista_reprobacion()
    return render_template('pro_reprobacion.html',lista = lista,ruta_grafica2=ruta_grafica2,ruta_grafica = ruta_grafica,m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

#funcion para realizar la proyeccion de desercion del modelo seleccionado -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/run_pro_deser", methods=['GET', 'POST'])
def run_pro_deser():
    select = request.form.get('lista_regresion')
    nombre_modelo = str(select)
    ruta_datos,ruta_grafica = pn.buscar_rutas_desercion(nombre_modelo)
    M = pn.cargar_datos_Selecionados_deser(ruta_datos)
    ruta_grafica2 = pn.Proyeccion_modelo_desercion(nombre_modelo)
    lista = pn.cargar_lista_desercion()
    return render_template('pro_desercion.html',lista = lista,ruta_grafica2=ruta_grafica2,ruta_grafica = ruta_grafica,m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])

#funcion para realizar la proyeccion de repitencia del modelo seleccionado -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/run_pro_repit", methods=['GET', 'POST'])
def run_pro_repit():
    select = request.form.get('lista_regresion')
    nombre_modelo = str(select)
    ruta_datos,ruta_grafica = pn.buscar_rutas_repitencia(nombre_modelo)
    M = pn.cargar_datos_Selecionados_repit(ruta_datos)
    ruta_grafica2 = pn.Proyeccion_modelo_repitencia(nombre_modelo)
    lista = pn.cargar_lista_repitencia()
    return render_template('pro_repitencia.html',lista = lista,ruta_grafica2=ruta_grafica2,ruta_grafica = ruta_grafica,m_coe=M[0],m_mse=M[1],m_ve=M[2],l_coe=M[3],l_mse=M[4],l_ve=M[5],r_coe=M[6],r_mse=M[7],r_ve=M[8])


if __name__ == "__main__":
    app.run(debug=True)
