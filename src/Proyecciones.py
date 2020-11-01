# librerias importadas para la realizacion de las regresiones lineales y sus respectivas proyecciones

import matplotlib.pyplot as plt # libreria para graficar funciones
import numpy as np # libreria procesos estadisticos
from sklearn.model_selection import train_test_split # libreria de entrenamiento de los datos
from sklearn.linear_model import LinearRegression,LassoCV, RidgeCV,Lasso,Ridge #libreria de los modelos de regresiones
from joblib import dump, load # libreria para guardar y cargar modelos de regresion
import pandas as pd # libreria de procesos estadisticos
from sklearn.metrics import mean_squared_error # libreria para hallar en minimo cuadrado
from numpy import savetxt, loadtxt # libreria para guardar y cargar archivios csv
import datetime # libreria para obtener fecha 
import os, shutil # librerias para aceder, crear y modificar directorios 

# funciones para obtener la rutas de los modelos selecionados y guardarlos en un lista ------------------------------------------------------------
def cargar_rutas_reprobacion():
    R  = loadtxt('Modelos/Rutas/ruta_reprobacion.csv', dtype="str", delimiter=',')
    return R

def cargar_rutas_desercion():
    R  = loadtxt('Modelos/Rutas/ruta_desercion.csv', dtype="str", delimiter=',')
    return R

def cargar_rutas_repitencia():
    R  = loadtxt('Modelos/Rutas/ruta_repitencia.csv', dtype="str", delimiter=',')
    return R

# funciones para cargar los datos de los modelos seleccionados y guardarlos en un lista --------------------------------------------------------------
def cargar_datos_Selecionados_repro(ruta_datos):
    M = loadtxt(ruta_datos+"/modelo_r_reprobacion.csv",dtype="str", delimiter=',')
    return M

def cargar_datos_Selecionados_deser(ruta_datos):
    M = loadtxt(ruta_datos+"/modelo_r_desercion.csv",dtype="str", delimiter=',')
    return M

def cargar_datos_Selecionados_repit(ruta_datos):
    M = loadtxt(ruta_datos+"/modelo_r_repitencia.csv",dtype="str", delimiter=',')
    return M

# funciones para obtener las direcciones de los datos y grafica en las rutas de los modelos seleccionados ----------------------------------------------
def buscar_rutas_reprobacion(nombre_modelo):
    R = cargar_rutas_reprobacion()
    indice = 0
    for i in range(0,len(R),1):
        if nombre_modelo in R[i]:
            indice = i 
    ruta_datos = R[indice+2]
    ruta_grafica = R[indice+3]
    ruta_grafica = ruta_grafica+"/regresion_reprobacion.png"
    return ruta_datos,ruta_grafica

def buscar_rutas_desercion(nombre_modelo):
    R = cargar_rutas_desercion()
    indice = 0
    for i in range(0,len(R),1):
        if nombre_modelo in R[i]:
            indice = i 
    ruta_datos = R[indice+2]
    ruta_grafica = R[indice+3]
    ruta_grafica = ruta_grafica+"/regresion_desercion.png"
    return ruta_datos,ruta_grafica

def buscar_rutas_repitencia(nombre_modelo):
    R = cargar_rutas_repitencia()
    indice = 0
    for i in range(0,len(R),1):
        if nombre_modelo in R[i]:
            indice = i 
    ruta_datos = R[indice+2]
    ruta_grafica = R[indice+3]
    ruta_grafica = ruta_grafica+"/regresion_repitencia.png"
    return ruta_datos,ruta_grafica


#  funciones para agregar un ruta a la lista de las rutas de los modelos seleccionados ------------------------------------------------------------
def cargar_lista_reprobacion():
    R  = cargar_rutas_reprobacion()
    lista = []
    for i in range(0,len(R),4):
        lista.append(R[i])
    return lista

def cargar_lista_desercion():
    R  = cargar_rutas_desercion()
    lista = []
    for i in range(0,len(R),4):
        lista.append(R[i])
    return lista

def cargar_lista_repitencia():
    R  = cargar_rutas_repitencia()
    lista = []
    for i in range(0,len(R),4):
        lista.append(R[i])
    return lista

# funciones para obtener la direccion del modelo en las rutas de los modelos seleccionados --------------------------------------------------------
def buscar_modelo_reprobacion(nombre_modelo):
    R = cargar_rutas_reprobacion()
    indice = 0
    for i in range(0,len(R),1):
        if nombre_modelo in R[i]:
            indice = i 
    ruta_modelo = R[indice+1]
    return ruta_modelo

def buscar_modelo_desercion(nombre_modelo):
    R = cargar_rutas_desercion()
    indice = 0
    for i in range(0,len(R),1):
        if nombre_modelo in R[i]:
            indice = i 
    ruta_modelo = R[indice+1]
    return ruta_modelo

def buscar_modelo_repitencia(nombre_modelo):
    R = cargar_rutas_repitencia()
    indice = 0
    for i in range(0,len(R),1):
        if nombre_modelo in R[i]:
            indice = i 
    ruta_modelo = R[indice+1]
    return ruta_modelo

# funciones para cargar los datos entrenados y guardarlos en un lista --------------------------------------------------
def cargar_datos_reporbacion():
    M = loadtxt('Modelos/Datos_entrenados/modelo_r_reprobacion.csv', delimiter=',')
    return M

def cargar_datos_desercion():
    M = loadtxt('Modelos/Datos_entrenados/modelo_r_desercion.csv', delimiter=',')
    return M

def cargar_datos_repitencia():
    M = loadtxt('Modelos/Datos_entrenados/modelo_r_repitencia.csv', delimiter=',')
    return M

# funciones para eliminar los modelos entrenados -------------------------------------------------------------------------
def elimiar_modelos_reprobacion():
    os.remove('Modelos/Entrenados/lr_reprobacion.pkl')
    os.remove('Modelos/Entrenados/rgl_reprobacion.pkl')
    os.remove('Modelos/Entrenados/rgr_reprobacion.pkl')
    
def elimiar_modelos_desercion():
    os.remove('Modelos/Entrenados/lr_desercion.pkl')
    os.remove('Modelos/Entrenados/rgl_desercion.pkl')
    os.remove('Modelos/Entrenados/rgr_desercion.pkl')

def elimiar_modelos_repitencia():
    os.remove('Modelos/Entrenados/lr_repitencia.pkl')
    os.remove('Modelos/Entrenados/rgl_repitencia.pkl')
    os.remove('Modelos/Entrenados/rgr_repitencia.pkl')


# funciones para guardar los rutas,modelos, datos y graficas seleccionados en sus respectivos directorios --------------------------------------------------------------------------------------------
def guardar_modelo_reprobacion():
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    lr = load('Modelos/Entrenados/lr_reprobacion.pkl')
    rgl = load('Modelos/Entrenados/rgl_reprobacion.pkl')
    rgr = load('Modelos/Entrenados/rgr_reprobacion.pkl')

    fecha =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dest_m = "Modelos/Seleccionados"
    dest_d = "Modelos/Datos_Seleccionados"
    dest_g = "static/file/seleccionados"
    dest_r = "Modelos/Rutas/ruta_reprobacion.csv"
    dir_m = os.path.join(dest_m,fecha)
    dir_d = os.path.join(dest_d,fecha)
    dir_g = os.path.join(dest_g,fecha)
    os.makedirs(dir_m)
    os.makedirs(dir_d)  
    os.makedirs(dir_g)

    shutil.copyfile("static/file/proyeccion_reprobacion_2.png",dir_g+"/regresion_reprobacion.png")
    
    dump(lr,dir_m+'/lr_reprobacion.pkl')
    dump(rgl,dir_m+'/rgl_reprobacion.pkl')
    dump(rgr,dir_m+'/rgr_reprobacion.pkl')

    datos = cargar_datos_reporbacion()
    savetxt(dir_d+'/modelo_r_reprobacion.csv', datos, fmt="%s" ,delimiter=',')

    if os.path.isfile(dest_r):
        ruta = loadtxt(dest_r, dtype="str", delimiter=',')
        ruta = np.append(ruta,["modelo_reprobacion - "+fecha,dir_m,dir_d,dir_g])
        
        savetxt(dest_r, ruta, fmt="%s" ,delimiter=',')
    else:
        ruta = ["modelo_reprobacion - "+fecha,dir_m,dir_d,dir_g]
        savetxt(dest_r, ruta, fmt="%s" ,delimiter=',')


def guardar_modelo_desercion():
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    lr = load('Modelos/Entrenados/lr_desercion.pkl')
    rgl = load('Modelos/Entrenados/rgl_desercion.pkl')
    rgr = load('Modelos/Entrenados/rgr_desercion.pkl')

    fecha =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dest_m = "Modelos/Seleccionados"
    dest_d = "Modelos/Datos_Seleccionados"
    dest_g = "static/file/seleccionados"
    dest_r = "Modelos/Rutas/ruta_desercion.csv"
    dir_m = os.path.join(dest_m,fecha)
    dir_d = os.path.join(dest_d,fecha)
    dir_g = os.path.join(dest_g,fecha)
    os.makedirs(dir_m)
    os.makedirs(dir_d)
    os.makedirs(dir_g)

    shutil.copyfile("static/file/proyeccion_desercion_2.png",dir_g+"/regresion_desercion.png")
    
    
    dump(lr,dir_m+'/lr_desercion.pkl')
    dump(rgl,dir_m+'/rgl_desercion.pkl')
    dump(rgr,dir_m+'/rgr_desercion.pkl')

    datos = cargar_datos_desercion()
    savetxt(dir_d+'/modelo_r_desercion.csv', datos, fmt="%s" ,delimiter=',')

    if os.path.isfile(dest_r):
        ruta = loadtxt(dest_r, dtype="str", delimiter=',')
        ruta = np.append(ruta,["modelo_desercion - "+fecha,dir_m,dir_d,dir_g])
        
        savetxt(dest_r, ruta, fmt="%s" ,delimiter=',')
    else:
        ruta = ["modelo_desercion - "+fecha,dir_m,dir_d,dir_g]
        savetxt(dest_r, ruta, fmt="%s" ,delimiter=',')




def guardar_modelo_repitencia():
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    lr = load('Modelos/Entrenados/lr_repitencia.pkl')
    rgl = load('Modelos/Entrenados/rgl_repitencia.pkl')
    rgr = load('Modelos/Entrenados/rgr_repitencia.pkl')

    fecha =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dest_m = "Modelos/Seleccionados"
    dest_d = "Modelos/Datos_Seleccionados"
    dest_g = "static/file/seleccionados"
    dir_m = os.path.join(dest_m,fecha)
    dir_d = os.path.join(dest_d,fecha)
    dir_g = os.path.join(dest_g,fecha)
    dest_r = "Modelos/Rutas/ruta_repitencia.csv"
    os.makedirs(dir_m)
    os.makedirs(dir_d)
    os.makedirs(dir_g)
    
    shutil.copyfile("static/file/proyeccion_repitencia_2.png",dir_g+"/regresion_repitencia.png")

    dump(lr,dir_m+'/lr_repitencia.pkl')
    dump(rgl,dir_m+'/rgl_repitencia.pkl')
    dump(rgr,dir_m+'/rgr_repitencia.pkl')

    datos = cargar_datos_repitencia()
    savetxt(dir_d+'/modelo_r_repitencia.csv', datos, fmt="%s" ,delimiter=',')

    if os.path.isfile(dest_r):
        ruta = loadtxt(dest_r, dtype="str", delimiter=',')
        ruta = np.append(ruta,["modelo_repitencia - "+fecha,dir_m,dir_d,dir_g])
        
        savetxt(dest_r, ruta, fmt="%s" ,delimiter=',')
    else:
        ruta = ["modelo_repitencia - "+fecha,dir_m,dir_d,dir_g]
        savetxt(dest_r, ruta, fmt="%s" ,delimiter=',')
 

# funciones de proyecciones de los modelos seleccionados en los soguientes 5 periodos academicos -------------------------------------------------------------------------------------------
def Proyeccion_modelo_reprobacion(nombre_modelo):
    ruta_modelo = buscar_modelo_reprobacion(nombre_modelo)
    ruta_grafica = "static/file/proyecciones/proyeccion_reprobacion.png"
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    lr = load(ruta_modelo+'/lr_reprobacion.pkl')
    rgl = load(ruta_modelo+'/rgl_reprobacion.pkl')
    rgr = load(ruta_modelo+'/rgr_reprobacion.pkl')
    año_actual =  datetime.datetime.now().year
    periodos =[año_actual,año_actual+1,año_actual+2,año_actual+3,año_actual+4]
    periodos = np.reshape(periodos, (-1, 1))
    y_pred = lr.predict(periodos)
    y_predrgl = rgl.predict(periodos)
    y_predrgr = rgr.predict(periodos)

    fig1 = plt.figure(figsize=(12,8), dpi=120)

    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pred, color='blue')
    ax.scatter(periodos,y_predrgl, color='yellow')
    ax.scatter(periodos,y_predrgr, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Reprobacion Media')
    fig1.savefig(ruta_grafica)
    return ruta_grafica
  
def Proyeccion_modelo_desercion(nombre_modelo):
    ruta_modelo = buscar_modelo_desercion(nombre_modelo)
    ruta_grafica = "static/file/proyecciones/proyeccion_desercion.png"
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    lr = load(ruta_modelo+'/lr_desercion.pkl')
    rgl = load(ruta_modelo+'/rgl_desercion.pkl')
    rgr = load(ruta_modelo+'/rgr_desercion.pkl')
    año_actual =  datetime.datetime.now().year
    periodos =[año_actual,año_actual+1,año_actual+2,año_actual+3,año_actual+4]
    periodos = np.reshape(periodos, (-1, 1))
    y_pred = lr.predict(periodos)
    y_predrgl = rgl.predict(periodos)
    y_predrgr = rgr.predict(periodos)

    fig1 = plt.figure(figsize=(12,8), dpi=120)

    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pred, color='blue')
    ax.scatter(periodos,y_predrgl, color='yellow')
    ax.scatter(periodos,y_predrgr, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Desercion Media')
    fig1.savefig(ruta_grafica)
    return ruta_grafica
  
def Proyeccion_modelo_repitencia(nombre_modelo):
    ruta_modelo = buscar_modelo_repitencia(nombre_modelo)
    ruta_grafica = "static/file/proyecciones/proyeccion_repitencia.png"
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    lr = load(ruta_modelo+'/lr_repitencia.pkl')
    rgl = load(ruta_modelo+'/rgl_repitencia.pkl')
    rgr = load(ruta_modelo+'/rgr_repitencia.pkl')
    año_actual =  datetime.datetime.now().year
    periodos =[año_actual,año_actual+1,año_actual+2,año_actual+3,año_actual+4]
    periodos = np.reshape(periodos, (-1, 1))
    y_pred = lr.predict(periodos)
    y_predrgl = rgl.predict(periodos)
    y_predrgr = rgr.predict(periodos)

    fig1 = plt.figure(figsize=(12,8), dpi=120)

    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pred, color='blue')
    ax.scatter(periodos,y_predrgl, color='yellow')
    ax.scatter(periodos,y_predrgr, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Repitencia Media')
    fig1.savefig(ruta_grafica)
    return ruta_grafica
  

# funciones de regresion por los metodos lineal, lasso y ridge -------------------------------------------- 
def proyeccion_reprobacion():
    datos=pd.read_csv('Datos/Datos_MEN.csv',header=0)
    # obtenemos la columna año de los datos
    datos['AÑO']=datos['AÑO'].astype('int64')
    # totamos solo los valos de la columna
    X=datos['AÑO'].values
    X=X[:, np.newaxis,]
    # obtenemos las columnas de reprobacion, desercion y repitencia de los datos
    y=datos['REPROBACIÓN_MEDIA'].values
    

    #separamos los datos de train en entrenamiento y prueba para probar los algoritmos

    # Reprobacion media
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

    # # Reprobacion media
    # lr= LinearRegression()
    # rgl = Lasso(alpha=.5)
    # rgr = Ridge(alpha=.5)

    # Se crean los modelos lineales
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    
    # #entreno el modelo

    # Reprobacion media
    lr.fit(X_train, y_train)
    rgl.fit(X_train, y_train)
    rgr.fit(X_train, y_train)

    # Guardar modelos
    dump(lr, 'Modelos/Entrenados/lr_reprobacion.pkl')
    dump(rgl, 'Modelos/Entrenados/rgl_reprobacion.pkl')
    dump(rgr, 'Modelos/Entrenados/rgr_reprobacion.pkl')

    # Reprobacion media
    y_pred = lr.predict(X_test)
    y_predrgl = rgl.predict(X_test)
    y_predrgr = rgr.predict(X_test)

    print('DATOS DEL MODELO DE REPROBACION')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    m_coe =lr.coef_
    print('Coeficientes:',lr.coef_)
    # MSE
    m_mse = "{0:.4f}".format(np.mean((lr.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    m_ve = "{0:.4f}".format(abs(lr.score(X_test,y_test)))
    
    print('Varianza explicada: %.2f\n' % lr.score(X_test,y_test))

    print('Regresión Lasso') 
    # Coeficiente
    l_coe =  rgl.coef_
    print('Coeficientes:', rgl.coef_)
    # MSE
    l_mse = "{0:.4f}".format(np.mean((rgl.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    l_ve = "{0:.4f}".format(abs(rgl.score(X_test, y_test)))
    print('Varianza explicada: %.2f\n' % rgl.score(X_test, y_test))

    print('Regresión Ridge')
    #Coeficiente
    r_coe = rgr.coef_
    print('Coeficientes:', rgr.coef_)
    # MSE
    r_mse = "{0:.4f}".format((np.mean(rgr.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    r_ve = "{0:.4f}".format(abs(rgr.score(X_test,y_test)))
    print('Varianza explicada: %.2f\n' % rgr.score(X_test,y_test))

    fig1 = plt.figure(figsize=(12,8), dpi=120)


    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(2, 1, 1)
    ax.scatter(X,y)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Reprobacion')
    ax.set_title('Grafica de Reprobacion')

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.scatter(X_test,y_test, color='black')
    ax2.plot(X_test, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión Lineal Reprobacion Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Prediccion Reprobacion Media')
    # plt.legend()
    # ax2.set_xticks(())
    # ax2.set_yticks(())
    fig1.savefig("static/file/proyeccion_reprobacion_2.png")
    # plt.show()

    m_coe2 = m_coe[0]
    l_coe2 = l_coe[0]
    r_coe2 = r_coe[0]

    modelo_reprobacion = [m_coe2,m_mse,m_ve,l_coe2,l_mse,l_ve,r_coe2,r_mse,r_ve]
    savetxt('Modelos/Datos_entrenados/modelo_r_reprobacion.csv', modelo_reprobacion, fmt="%s" ,delimiter=',')

    return m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve


def proyeccion_desercion():
    datos=pd.read_csv('Datos/Datos_MEN.csv',header=0)
    # obtenemos la columna año de los datos
    datos['AÑO']=datos['AÑO'].astype('int64')
    # totamos solo los valos de la columna
    X=datos['AÑO'].values
    X=X[:, np.newaxis,]
    # obtenemos las columnas de reprobacion, desercion y repitencia de los datos
    y2=datos['DESERCIÓN_MEDIA'].values

    #separamos los datos de train en entrenamiento y prueba para probar los algoritmos

    # Desercion media
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2,test_size=0.5)

    # definicion del algoritmo a utilizar

    # # Desercion media
    # lr2= LinearRegression()
    # rgl2 = Lasso(alpha=.5)
    # rgr2 = Ridge(alpha=.5)

    # Se crean los modelos lineales
    lr2 = LinearRegression()
    rgl2 = LassoCV(cv=4)
    rgr2 = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])

    # #entreno el modelo

    # Desercion media
    lr2.fit(X_train2, y_train2)
    rgl2.fit(X_train2, y_train2)
    rgr2.fit(X_train2, y_train2)

    # Guardar Modelos
    dump(lr2, 'Modelos/Entrenados/lr_desercion.pkl')
    dump(rgl2, 'Modelos/Entrenados/rgl_desercion.pkl')
    dump(rgr2, 'Modelos/Entrenados/rgr_desercion.pkl')

    #realizamos la prediccion

    #Desercion media
    y_pred2 = lr2.predict(X_test2)
    y_predrgl2 = rgl2.predict(X_test2)
    y_predrgr2 = rgr2.predict(X_test2)

    #  ------------------------------------------------------------------------------------------

    print('DATOS DEL MODELO DE DESERCION')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    m_coe = lr2.coef_
    print('Coeficientes:',lr2.coef_)
    # MSE
    m_mse = "{0:.4f}".format(np.mean((lr2.predict(X_test2) - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr2.predict(X_test2) - y_test2) ** 2))
    # Varianza explicada
    m_ve = "{0:.4f}".format(abs(lr2.score(X_test2,y_test2)))
    print('Varianza explicada: %.2f\n' % lr2.score(X_test2,y_test2))

    print('Regresión Lasso') 
    # Coeficiente
    l_coe = rgl2.coef_
    print('Coeficientes:', rgl2.coef_)
    # MSE
    l_mse = "{0:.4f}".format(np.mean((rgl2.predict(X_test2) - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl2.predict(X_test2) - y_test2) ** 2))
    # Varianza explicada
    l_ve = "{0:.4f}".format(abs(rgl2.score(X_test2, y_test2)))
    print('Varianza explicada: %.2f\n' % rgl2.score(X_test2, y_test2))

    print('Regresión Ridge')
    #Coeficiente
    r_coe = rgr2.coef_
    print('Coeficientes:', rgr2.coef_)
    # MSE
    r_mse = "{0:.4f}".format(np.mean((rgr2.predict(X_test2) - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr2.predict(X_test2) - y_test2) ** 2))
    # Varianza explicada
    r_ve = "{0:.4f}".format(abs(rgr2.score(X_test2,y_test2)))
    print('Varianza explicada: %.2f\n' % rgr2.score(X_test2,y_test2))

    fig2 = plt.figure(figsize=(12,8), dpi=120)


    fig2.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig2.add_subplot(2, 1, 1)
    ax.scatter(X,y2)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Desercion')
    ax.set_title('Grafica de Desercion')
  

    ax2 = fig2.add_subplot(2, 1, 2)
    ax2.scatter(X_test2,y_test2, color='black')
    ax2.plot(X_test2, y_pred2, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test2, y_predrgl2, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test2, y_predrgr2, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión Lineal Desercion Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Prediccion Desercion Media')
    fig2.savefig("static/file/proyeccion_desercion_2.png")
    # plt.show()
    m_coe2 = m_coe[0]
    l_coe2 = l_coe[0]
    r_coe2 = r_coe[0]

    modelo_desercion = [m_coe2,m_mse,m_ve,l_coe2,l_mse,l_ve,r_coe2,r_mse,r_ve]
    savetxt('Modelos/Datos_entrenados/modelo_r_desercion.csv', modelo_desercion,fmt="%s" , delimiter=',')

    return m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve


def proyeccion_repitencia():
    datos=pd.read_csv('Datos/Datos_MEN.csv',header=0)
    # obtenemos la columna año de los datos
    datos['AÑO']=datos['AÑO'].astype('int64')
    # totamos solo los valos de la columna
    X=datos['AÑO'].values
    X=X[:, np.newaxis,]
    # obtenemos las columnas de reprobacion, desercion y repitencia de los datos
    y3=datos['REPITENCIA_MEDIA'].values

    #separamos los datos de train en entrenamiento y prueba para probar los algoritmos

    # Repitencia media
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3,test_size=0.5)

    #definicion del algoritmo a utilizar

    # # Repitencia media
    # lr3 = LinearRegression()
    # rgl3 = Lasso(alpha=.5)
    # rgr3 = Ridge(alpha=.5)

    # Se crean los modelos lineales
    lr3 = LinearRegression()
    rgl3 = LassoCV(cv=4)
    rgr3 = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])


    # #entreno el modelo

    # Repitencia media
    lr3.fit(X_train3, y_train3)
    rgl3.fit(X_train3, y_train3)
    rgr3.fit(X_train3, y_train3)


    # Guardar Modelos
    dump(lr3, 'Modelos/Entrenados/lr_repitencia.pkl')
    dump(rgl3, 'Modelos/Entrenados/rgl_repitencia.pkl')
    dump(rgr3, 'Modelos/Entrenados/rgr_repitencia.pkl')

    #realizamos la prediccion

    #Repitencia media
    y_pred3 = lr3.predict(X_test3)
    y_predrgl3 = rgl3.predict(X_test3)
    y_predrgr3 = rgr3.predict(X_test3)

    #  ------------------------------------------------------------------------------------------

    print('DATOS DEL MODELO DE REPITENCIA')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    m_coe = lr3.coef_
    print('Coeficientes:',lr3.coef_)
    # MSE
    m_mse = "{0:.4f}".format(np.mean((lr3.predict(X_test3) - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr3.predict(X_test3) - y_test3) ** 2))
    # Varianza explicada
    m_ve = "{0:.4f}".format(abs(lr3.score(X_test3,y_test3)))
    print('Varianza explicada: %.2f\n' % lr3.score(X_test3,y_test3))

    print('Regresión Lasso') 
    # Coeficiente
    l_coe =  rgl3.coef_
    print('Coeficientes:', rgl3.coef_)
    # MSE
    l_mse = "{0:.4f}".format((np.mean(rgl3.predict(X_test3) - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl3.predict(X_test3) - y_test3) ** 2))
    # Varianza explicada
    l_ve = "{0:.4f}".format(abs(rgl3.score(X_test3, y_test3)))
    print('Varianza explicada: %.2f\n' % rgl3.score(X_test3, y_test3))

    print('Regresión Ridge')
    #Coeficiente
    r_coe = rgr3.coef_
    print('Coeficientes:', rgr3.coef_)
    # MSE
    r_mse = "{0:.4f}".format((np.mean(rgr3.predict(X_test3) - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr3.predict(X_test3) - y_test3) ** 2))
    # Varianza explicada
    r_ve = "{0:.4f}".format(abs(rgr3.score(X_test3,y_test3)))
    print('Varianza explicada: %.2f\n' % rgr3.score(X_test3,y_test3))


    fig3 = plt.figure(figsize=(12,8), dpi=120)


    fig3.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig3.add_subplot(2, 1, 1)
    ax.scatter(X,y3)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Repitencia')
    ax.set_title('Grafica de Repitencia')

    ax2 = fig3.add_subplot(2, 1, 2)
    ax2.scatter(X_test3,y_test3, color='black')
    ax2.plot(X_test3, y_pred3, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test3, y_predrgl3, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test3, y_predrgr3, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión Lineal Repitencia Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Prediccion Repitencia Media')
    fig3.savefig("static/file/proyeccion_repitencia_2.png")
    # plt.show()
    m_coe2 = m_coe[0]
    l_coe2 = l_coe[0]
    r_coe2 = r_coe[0]

    modelo_repitencia = [m_coe2,m_mse,m_ve,l_coe2,l_mse,l_ve,r_coe2,r_mse,r_ve]
    savetxt('Modelos/Datos_entrenados/modelo_r_repitencia.csv', modelo_repitencia,fmt="%s", delimiter=',')

    return m_coe,m_mse,m_ve,l_coe,l_mse,l_ve,r_coe,r_mse,r_ve
    

