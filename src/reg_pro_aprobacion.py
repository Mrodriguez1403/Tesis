import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from numpy import savetxt, loadtxt
import datetime


def reg_pro_aprobacion():
    datos=pd.read_csv('Datos/Datos_MEN.csv',header=0)
    # obtenemos la columna año de los datos
    datos['AÑO']=datos['AÑO'].astype('int64')
    # totamos solo los valos de la columna
    X=datos['AÑO'].values
    X=X[:, np.newaxis,]
    # obtenemos las columna de Aprobacion de los datos
    y=datos['APROBACIÓN_MEDIA'].values

    # Aprobacion media
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

     # Se crean los modelos lineales
    lr = LinearRegression()
    rgl = LassoCV(cv=4)
    rgr = RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
    
    # #entreno el modelo

    # Aprobacion media
    lr.fit(X_train, y_train)
    rgl.fit(X_train, y_train)
    rgr.fit(X_train, y_train)

    # realizamos la prediccion
    # Aprobacion media
    y_pred = lr.predict(X_test)
    y_predrgl = rgl.predict(X_test)
    y_predrgr = rgr.predict(X_test)

    # periodos a proyectar -----------------------------------------------------------------------------
    año_actual =  datetime.datetime.now().year
    periodos =[año_actual,año_actual+1,año_actual+2,año_actual+3,año_actual+4]
    periodos = np.reshape(periodos, (-1, 1))

    # Aprobacion media
    y_pro2 = lr.predict(periodos)
    y_prorgl2 = rgl.predict(periodos)
    y_prorgr2 = rgr.predict(periodos)

    print('DATOS DEL MODELO DE REPROBACION')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    # m_coe =lr.coef_
    print('Coeficientes:',lr.coef_)
    # MSE
    # m_mse = "{0:.4f}".format(np.mean((lr.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    # m_ve = "{0:.4f}".format(abs(lr.score(X_test,y_test)))
    
    print('Varianza explicada: %.2f\n' % lr.score(X_test,y_test))

    print('Regresión Lasso') 
    # Coeficiente
    # l_coe =  rgl.coef_
    print('Coeficientes:', rgl.coef_)
    # MSE
    # l_mse = "{0:.4f}".format(np.mean((rgl.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    # l_ve = "{0:.4f}".format(abs(rgl.score(X_test, y_test)))
    print('Varianza explicada: %.2f\n' % rgl.score(X_test, y_test))

    print('Regresión Ridge')
    #Coeficiente
    # r_coe = rgr.coef_
    print('Coeficientes:', rgr.coef_)
    # MSE
    # r_mse = "{0:.4f}".format((np.mean(rgr.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    # r_ve = "{0:.4f}".format(abs(rgr.score(X_test,y_test)))
    print('Varianza explicada: %.2f\n' % rgr.score(X_test,y_test))

    fig1 = plt.figure(figsize=(12,8), dpi=120)


    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(2, 1, 1)
    ax.scatter(X,y)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Aprobación')
    ax.set_title('Grafica de Aprobación')

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.scatter(X_test,y_test, color='black')
    ax2.plot(X_test, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión de Aprobación Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Regresión de Aprobación Media')
    # plt.legend()
    # ax2.set_xticks(())
    # ax2.set_yticks(())
    ruta_reg = "static/file/regresion_aprobacion.png"
    fig1.savefig(ruta_reg)
    # plt.show()

    fig01 = plt.figure(figsize=(12,8), dpi=120)

    fig01.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig01.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pro2, color='blue')
    ax.scatter(periodos,y_prorgl2, color='yellow')
    ax.scatter(periodos,y_prorgr2, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Aprobación Media')
    ruta_pro = "static/file/pro_aprobacion.png"
    fig01.savefig(ruta_pro)


    return ruta_reg,ruta_pro
