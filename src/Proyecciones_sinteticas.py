import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from numpy import savetxt, loadtxt
import datetime
# -------------------------------------------------------------------------------

def proyecciones_sinteticas():
    X = []
    X = loadtxt('Datos/periodos.csv', delimiter=',')
    X = X[:, np.newaxis, ]
    y = []
    y = loadtxt('Datos/datos_Reprobacion.csv', delimiter=',')
    y2 = []
    y2 = loadtxt('Datos/datos_Desercion.csv', delimiter=',')
    y3 = []
    y3 = loadtxt('Datos/datos_Repitencia.csv', delimiter=',')


    #separamos los datos de train en entrenamiento y prueba para probar los algoritmos

    # Reprobacion media
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

    # Desercion media
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2,test_size=0.5)

    # Repitencia media
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3,test_size=0.5)

    # definicion del algoritmo a utilizar -----------------------------------------------------------------------

    # Reprobacion media
    lr= LinearRegression()
    rgl = Lasso(alpha=.5)
    rgr = Ridge(alpha=.5)

    # Desercion media
    lr2= LinearRegression()
    rgl2 = Lasso(alpha=.5)
    rgr2 = Ridge(alpha=.5)

    # Repitencia media
    lr3= LinearRegression()
    rgl3 = Lasso(alpha=.5)
    rgr3 = Ridge(alpha=.5)

    # #entreno el modelo  ----------------------------------------------------------------------------------

    # Reprobacion media
    lr.fit(X_train, y_train)
    rgl.fit(X_train, y_train)
    rgr.fit(X_train, y_train)

    # Desercion media
    lr2.fit(X_train2, y_train2)
    rgl2.fit(X_train2, y_train2)
    rgr2.fit(X_train2, y_train2)

    # Repitencia media
    lr3.fit(X_train3, y_train3)
    rgl3.fit(X_train3, y_train3)
    rgr3.fit(X_train3, y_train3)



    #realizamos la prediccion  ---------------------------------------------------------------------------------

    # Reprobacion media
    y_pred = lr.predict(X_test)
    y_predrgl = rgl.predict(X_test)
    y_predrgr = rgr.predict(X_test)

    # Desercion media
    y_pred2 = lr2.predict(X_test2)
    y_predrgl2 = rgl2.predict(X_test2)
    y_predrgr2 = rgr2.predict(X_test2)

    # Repitencia media
    y_pred3 = lr3.predict(X_test3)
    y_predrgl3 = rgl3.predict(X_test3)
    y_predrgr3 = rgr3.predict(X_test3)

    # periodos a proyectar -----------------------------------------------------------------------------
    año_actual =  datetime.datetime.now().year
    periodos =[año_actual,año_actual+1,año_actual+2,año_actual+3,año_actual+4]
    periodos = np.reshape(periodos, (-1, 1))

 #realizamos la proyecciones  ---------------------------------------------------------------------------------

    # Reprobacion media
    y_pro = lr.predict(periodos)
    y_prorgl = rgl.predict(periodos)
    y_prorgr = rgr.predict(periodos)

    # Desercion media
    y_pro2 = lr2.predict(periodos)
    y_prorgl2 = rgl2.predict(periodos)
    y_prorgr2 = rgr2.predict(periodos)

    # Repitencia media
    y_pro3 = lr3.predict(periodos)
    y_prorgl3 = rgl3.predict(periodos)
    y_prorgr3 = rgr3.predict(periodos)

    # generar graficas ---------------------------------------------------------------------------------

    fig1 = plt.figure(figsize=(12,8), dpi=120)

    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(2, 1, 1)
    ax.scatter(X,y)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Reprobacion')
    ax.set_title('Grafica de Reprobacion')

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.scatter(X_test,y_test)
    ax2.plot(X_test, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión Reprobacion Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Regresion Reprobacion Media')

    print('DATOS DEL MODELO DE REPROBACION')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    # m_coe =lr.coef_
    print('Coeficientes:',lr.coef_)
    # MSE
    # m_mse = "{0:.4f}".format(np.mean((lr.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    # m_ve = "{0:.4f}".format(lr.score(X_test,y_test))
    print('Varianza explicada: %.2f\n' % lr.score(X_test,y_test))

    print('Regresión Lasso') 
    # Coeficiente
    # l_coe =  rgl.coef_
    print('Coeficientes:', rgl.coef_)
    # MSE
    # l_mse = "{0:.4f}".format(np.mean((rgl.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    # l_ve = "{0:.4f}".format(rgl.score(X_test, y_test))
    print('Varianza explicada: %.2f\n' % rgl.score(X_test, y_test))

    print('Regresión Ridge')
    #Coeficiente
    # r_coe = rgr.coef_
    print('Coeficientes:', rgr.coef_)
    # MSE
    # r_mse = "{0:.4f}".format(np.mean((rgr.predict(X_test) - y_test) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr.predict(X_test) - y_test) ** 2))
    # Varianza explicada
    # r_ve = "{0:.4f}".format(rgr.score(X_test,y_test))
    print('Varianza explicada: %.2f\n' % rgr.score(X_test,y_test))

    ruta_reprobacion = "static/file/reg_s_reprobacion.png"
    fig1.savefig(ruta_reprobacion)
    

    fig01 = plt.figure(figsize=(12,8), dpi=120)

    fig01.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig01.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pro, color='blue')
    ax.scatter(periodos,y_prorgl, color='yellow')
    ax.scatter(periodos,y_prorgr, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Reprobacion Media')
    ruta_pro_repro = "static/file/pro_s_reprobacion.png"
    fig01.savefig(ruta_pro_repro)

    # plt.show()

    #  ------------------------------------------------------------------------------------------

    fig2 = plt.figure(figsize=(12,8), dpi=120)


    fig2.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig2.add_subplot(2, 1, 1)
    ax.scatter(X,y2)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Desercion')
    ax.set_title('Grafica de Desercion')

    ax2 = fig2.add_subplot(2, 1, 2)
    ax2.scatter(X_test2,y_test2)
    ax2.plot(X_test2, y_pred2, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test2, y_predrgl2, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test2, y_predrgr2, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión Desercion Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Regresión Dersercion Media')

    print('DATOS DEL MODELO DE DESERCION SINTETICA')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    # m_coe2 =lr2.coef_
    print('Coeficientes:',lr2.coef_)
    # MSE
    # m_mse2 = "{0:.4f}".format(np.mean((lr2.predict(X_test2= - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr2.predict(X_test2) - y_test2) ** 2))
    # Varianza explicada
    # m_ve2 = "{0:.4f}".format(lr2.score(X_test2,y_test2))
    print('Varianza explicada: %.2f\n' % lr2.score(X_test2,y_test2))

    print('Regresión Lasso') 
    # Coeficiente
    # l_coe2 =  rgl2.coef_
    print('Coeficientes:', rgl2.coef_)
    # MSE
    # l_mse2 = "{0:.4f}".format(np.mean(rgl2.predict(X_test2 - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl2.predict(X_test2) - y_test2) ** 2))
    # Varianza explicada
    # l_ve2 = "{0:.4f}".format(rgl2.score(X_test2, y_test2))
    print('Varianza explicada: %.2f\n' % rgl2.score(X_test2, y_test2))

    print('Regresión Ridge')
    #Coeficiente
    # r_coe2 = rgr2.coef_
    print('Coeficientes:', rgr2.coef_)
    # MSE
    # r_mse2 = "{0:.4f}".format(np.mean((rgr2.predict(X_test2= - y_test2) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr2.predict(X_test2) - y_test2) ** 2))
    # Varianza explicada
    # r_ve2 = "{0:.4f}".format(rgr2.score(X_test2,y_test2))
    print('Varianza explicada: %.2f\n' % rgr2.score(X_test2,y_test2))

    ruta_desercion="static/file/reg_s_desercion.png"
    fig2.savefig(ruta_desercion)


    fig02 = plt.figure(figsize=(12,8), dpi=120)

    fig02.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig02.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pro2, color='blue')
    ax.scatter(periodos,y_prorgl2, color='yellow')
    ax.scatter(periodos,y_prorgr2, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Reprobacion Media')
    ruta_pro_deser = "static/file/pro_s_desercion.png"
    fig02.savefig(ruta_pro_deser)
    # plt.show()


    #  ------------------------------------------------------------------------------------------

    fig3 = plt.figure(figsize=(12,8), dpi=120)

    fig3.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig3.add_subplot(2, 1, 1)
    ax.scatter(X,y3)
    ax.set_xlabel('Periodos') 
    ax.set_ylabel('Tasa de Repitencia')
    ax.set_title('Grafica de Repitencia')

    ax2 = fig3.add_subplot(2, 1, 2)
    ax2.scatter(X_test3,y_test3)
    ax2.plot(X_test3, y_pred3, color='blue',linewidth=3, label=u'Regresión MCO')
    ax2.plot(X_test3, y_predrgl3, color='yellow',linewidth=3, label=u'Regresión Lasso')
    ax2.plot(X_test3, y_predrgr3, color='green',linewidth=3, label=u'Regresión Ridge')
    ax2.set_title(u'Regresión Repitencia Media por 3 metodos diferentes')
    ax2.set_xlabel('Periodos')
    ax2.set_ylabel('Regresión Reprobacion Media')

    print('DATOS DEL MODELO DE REPRITENCIA SINTETICA')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    # m_coe3 =lr3.coef_
    print('Coeficientes:',lr3.coef_)
    # MSE
    # m_mse3 = "{0:.4f}".format(np.mean((lr3.predict(X_test3) - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((lr3.predict(X_test3) - y_test3) ** 2))
    # Varianza explicada
    # m_ve3 = "{0:.4f}".format(lr3.score(X_test3,y_test3))
    print('Varianza explicada: %.2f\n' % lr3.score(X_test3,y_test3))

    print('Regresión Lasso') 
    # Coeficiente
    # l_coe3 =  rgl3.coef_
    print('Coeficientes:', rgl3.coef_)
    # MSE
    # l_mse3 = "{0:.4f}".format(np.mean((rgl3.predict(X_test3) - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgl3.predict(X_test3) - y_test3) ** 2))
    # Varianza explicada
    # l_ve3 = "{0:.4f}".format(rgl3.score(X_test3, y_test3))
    print('Varianza explicada: %.2f\n' % rgl3.score(X_test3, y_test3))

    print('Regresión Ridge')
    #Coeficiente
    # r_coe3 = rgr3.coef_
    print('Coeficientes:', rgr3.coef_)
    # MSE
    # r_mse3 = "{0:.4f}".format(np.mean((rgr3.predict(X_test3) - y_test3) ** 2))
    print("Residual sum of squares: %.2f"% np.mean((rgr3.predict(X_test3) - y_test3) ** 2))
    # Varianza explicada
    # r_ve3 = "{0:.4f}".format(rgr3.score(X_test3,y_test3))
    print('Varianza explicada: %.2f\n' % rgr3.score(X_test3,y_test3))


    ruta_repitencia = "static/file/reg_s_repitencia.png"
    fig3.savefig(ruta_repitencia)

    fig03 = plt.figure(figsize=(12,8), dpi=120)

    fig03.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig03.add_subplot(1, 1, 1)
    ax.scatter(periodos,y_pro3, color='blue')
    ax.scatter(periodos,y_prorgl3, color='yellow')
    ax.scatter(periodos,y_prorgr3, color='green')
    # ax2.plot(periodos, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # ax2.plot(periodos, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # ax2.plot(periodos, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    ax.set_title(u'Proyeccion de los siguentes 5 Periodos Escolares')
    ax.set_xlabel('Periodos')
    ax.set_ylabel('Proyeccion Reprobacion Media')
    ruta_pro_repit = "static/file/pro_s_repitencia.png"
    fig03.savefig(ruta_pro_repit)

    # datos = [m_coe,m_coe2,m_coe3,m_mse,m_mse2,m_mse3,m_ve,m_ve2,m_ve3,l_coe,l_coe2,l_coe3,l_mse,l_mse2,l_mse3,l_ve,l_ve2,l_ve3,r_coe,r_coe2,r_coe3,r_mse,r_mse2,r_mse3,r_ve,r_ve2,r_ve3]
   
    return ruta_reprobacion,ruta_desercion,ruta_repitencia,ruta_pro_repro,ruta_pro_deser,ruta_pro_repit
    # plt.show()




