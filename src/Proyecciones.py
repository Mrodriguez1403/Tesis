import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso, Ridge
from joblib import dump, load
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import savetxt, loadtxt
import datetime
import os, shutil

def cargar_datos_reporbacion():
    M = loadtxt('Modelos/Datos_entrenados/modelo_r_reprobacion.csv', delimiter=',')
    return M

def cargar_datos_desercion():
    M = loadtxt('Modelos/Datos_entrenados/modelo_r_desercion.csv', delimiter=',')
    return M

def cargar_datos_repitencia():
    M = loadtxt('Modelos/Datos_entrenados/modelo_r_repitencia.csv', delimiter=',')
    return M

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



def guardar_modelo_reprobacion():
    lr= LinearRegression()
    rgl = Lasso(alpha=.5)
    rgr = Ridge(alpha=.5)
    lr = load('Modelos/Entrenados/lr_reprobacion.pkl')
    rgl = load('Modelos/Entrenados/rgl_reprobacion.pkl')
    rgr = load('Modelos/Entrenados/rgr_reprobacion.pkl')

    fecha =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dest_m = "Modelos/Seleccionados"
    dest_d = "Modelos/Datos_Seleccionados"
    dest_g = "static/file/seleccionados"
    dest_r = "Modelos/Rutas"
    dir_m = os.path.join(dest_m,fecha)
    dir_d = os.path.join(dest_d,fecha)
    dir_g = os.path.join(dest_g,fecha)
    dir_r = os.path.join(dest_r,fecha)
    os.makedirs(dir_m)
    os.makedirs(dir_d)  
    os.makedirs(dir_g)
    os.makedirs(dir_r)

    shutil.copyfile("static/file/proyeccion_reprobacion_2.png",dir_g+"/regresion_reprobacion.png")
    
    dump(lr,dir_m+'/lr_reprobacion.pkl')
    dump(rgl,dir_m+'/rgl_reprobacion.pkl')
    dump(rgr,dir_m+'/rgr_reprobacion.pkl')

    datos = cargar_datos_reporbacion()
    savetxt(dir_d+'/modelo_r_reprobacion.csv', datos, fmt="%s" ,delimiter=',')

    ruta = ["",dir_m,dir_d,dir_g,fecha]
    savetxt(dir_r+"/ruta_reprobacion.csv", ruta, fmt="%s" ,delimiter=',')
    

    

def guardar_modelo_desercion():
    lr= LinearRegression()
    rgl = Lasso(alpha=.5)
    rgr = Ridge(alpha=.5)
    lr = load('Modelos/Entrenados/lr_desercion.pkl')
    rgl = load('Modelos/Entrenados/rgl_desercion.pkl')
    rgr = load('Modelos/Entrenados/rgr_desercion.pkl')

    fecha =  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dest_m = "Modelos/Seleccionados"
    dest_d = "Modelos/Datos_Seleccionados"
    dest_g = "static/file/seleccionados"
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




def guardar_modelo_repitencia():
    lr= LinearRegression()
    rgl = Lasso(alpha=.5)
    rgr = Ridge(alpha=.5)
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
    os.makedirs(dir_m)
    os.makedirs(dir_d)
    os.makedirs(dir_g)
    
    shutil.copyfile("static/file/proyeccion_repitencia_2.png",dir_g+"/regresion_repitencia.png")

    dump(lr,dir_m+'/lr_repitencia.pkl')
    dump(rgl,dir_m+'/rgl_repitencia.pkl')
    dump(rgr,dir_m+'/rgr_repitencia.pkl')

    datos = cargar_datos_repitencia()
    savetxt(dir_d+'/modelo_r_repitencia.csv', datos, fmt="%s" ,delimiter=',')
 


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

    # Reprobacion media
    lr= LinearRegression()
    rgl = Lasso(alpha=.5)
    rgr = Ridge(alpha=.5)
    
    # Guardar modelos
    dump(lr, 'Modelos/Entrenados/lr_reprobacion.pkl')
    dump(rgl, 'Modelos/Entrenados/rgl_reprobacion.pkl')
    dump(rgr, 'Modelos/Entrenados/rgr_reprobacion.pkl')

    # #entreno el modelo

    # Reprobacion media
    lr.fit(X_train, y_train)
    rgl.fit(X_train, y_train)
    rgr.fit(X_train, y_train)

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
    m_mse = "{0:.4f}".format(np.mean((y_pred - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_pred - y_test) ** 2))
    # Varianza explicada
    m_ve = "{0:.4f}".format(lr.score(X_test,y_test))
    print('Varianza explicada: %.2f\n' % lr.score(X_test,y_test))

    print('Regresión Lasso') 
    # Coeficiente
    l_coe =  rgl.coef_
    print('Coeficientes:', rgl.coef_)
    # MSE
    l_mse = "{0:.4f}".format(np.mean((y_predrgl - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_predrgl - y_test) ** 2))
    # Varianza explicada
    l_ve = "{0:.4f}".format(rgl.score(X_test, y_test))
    print('Varianza explicada: %.2f\n' % rgl.score(X_test, y_test))

    print('Regresión Ridge')
    #Coeficiente
    r_coe = rgr.coef_
    print('Coeficientes:', rgr.coef_)
    # MSE
    r_mse = "{0:.4f}".format(np.mean((y_predrgr - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_predrgr - y_test) ** 2))
    # Varianza explicada
    r_ve = "{0:.4f}".format(rgr.score(X_test,y_test))
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

    # Desercion media
    lr2= LinearRegression()
    rgl2 = Lasso(alpha=.5)
    rgr2 = Ridge(alpha=.5)

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
    m_mse = "{0:.4f}".format(np.mean((y_pred2 - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_pred2 - y_test2) ** 2))
    # Varianza explicada
    m_ve = "{0:.4f}".format(lr2.score(X_test2,y_test2))
    print('Varianza explicada: %.2f\n' % lr2.score(X_test2,y_test2))

    print('Regresión Lasso') 
    # Coeficiente
    l_coe = rgl2.coef_
    print('Coeficientes:', rgl2.coef_)
    # MSE
    l_mse = "{0:.4f}".format(np.mean((y_predrgl2 - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_predrgl2 - y_test2) ** 2))
    # Varianza explicada
    l_ve = "{0:.4f}".format(rgl2.score(X_test2, y_test2))
    print('Varianza explicada: %.2f\n' % rgl2.score(X_test2, y_test2))

    print('Regresión Ridge')
    #Coeficiente
    r_coe = rgr2.coef_
    print('Coeficientes:', rgr2.coef_)
    # MSE
    r_mse = "{0:.4f}".format(np.mean((y_predrgr2 - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_predrgr2 - y_test2) ** 2))
    # Varianza explicada
    r_ve = "{0:.4f}".format(rgr2.score(X_test2,y_test2))
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

    # Repitencia media
    lr3= LinearRegression()
    rgl3 = Lasso(alpha=.5)
    rgr3 = Ridge(alpha=.5)

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
    m_mse = "{0:.4f}".format(np.mean((y_pred3 - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_pred3 - y_test3) ** 2))
    # Varianza explicada
    m_ve = "{0:.4f}".format(lr3.score(X_test3,y_test3))
    print('Varianza explicada: %.2f\n' % lr3.score(X_test3,y_test3))

    print('Regresión Lasso') 
    # Coeficiente
    l_coe =  rgl3.coef_
    print('Coeficientes:', rgl3.coef_)
    # MSE
    l_mse = "{0:.4f}".format(np.mean((y_predrgl3 - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_predrgl3 - y_test3) ** 2))
    # Varianza explicada
    l_ve = "{0:.4f}".format(rgl3.score(X_test3, y_test3))
    print('Varianza explicada: %.2f\n' % rgl3.score(X_test3, y_test3))

    print('Regresión Ridge')
    #Coeficiente
    r_coe = rgr3.coef_
    print('Coeficientes:', rgr3.coef_)
    # MSE
    r_mse = "{0:.4f}".format(np.mean((y_predrgr3 - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((y_predrgr3 - y_test3) ** 2))
    # Varianza explicada
    r_ve = "{0:.4f}".format(rgr3.score(X_test3,y_test3))
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
    

