import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso, Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import savetxt


# -------------------------------------------------------------------------------
def proyecciones_reales():
    datos=pd.read_csv('Datos/Datos_MEN.csv',header=0)
    # obtenemos la columna año de los datos
    datos['AÑO']=datos['AÑO'].astype('int64')
    # totamos solo los valos de la columna
    X=datos['AÑO'].values
    X=X[:, np.newaxis,]
    # obtenemos las columnas de reprobacion, desercion y repitencia de los datos
    y=datos['REPROBACIÓN_MEDIA'].values
    y2=datos['DESERCIÓN_MEDIA'].values
    y3=datos['REPITENCIA_MEDIA'].values



    #separamos los datos de train en entrenamiento y prueba para probar los algoritmos

    # Reprobacion media
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

    # Desercion media
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2,test_size=0.5)

    # Repitencia media
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3,test_size=0.5)

    # definicion del algoritmo a utilizar

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

    # #entreno el modelo

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


    #realizamos la prediccion

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


    #  ------------------------------------------------------------------------------------------
    print('DATOS DEL MODELO DE REPROBACION')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    print('Coeficientes:',lr.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_pred - y_test) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % lr.score(X_test,y_test))

    print('Regresión Lasso') 
    # Coeficiente
    print('Coeficientes:', rgl.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_predrgl - y_test) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % rgl.score(X_test, y_test))

    print('Regresión Ridge')
    #Coeficiente
    print('Coeficientes:', rgr.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_predrgr - y_test) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % rgr.score(X_test,y_test))


    # plt.scatter(X_test,y_test, color='black')
    # plt.plot(X_test, y_pred, color='blue',linewidth=3, label=u'Regresión MCO')
    # plt.plot(X_test, y_predrgl, color='yellow',linewidth=3, label=u'Regresión Lasso')
    # plt.plot(X_test, y_predrgr, color='green',linewidth=3, label=u'Regresión Ridge')
    # plt.title(u'Regresión Lineal Reprobacion Media por 3 metodos diferentes')
    # plt.legend()
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()


    # creamos una figura que guardara la grafica de reprobacion 
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

    # # definimos los datos relevantes del modelo de reprobacion
    # print('DATOS DEL MODELO DE REPROBACION')
    # print()
    # print('Valor de la pendiente o coeficiente "a":')
    # a = lr.coef_
    # print(a)
    # print('Valor de la intersección o coeficiente "b":')
    # b = lr.intercept_
    # print(b)
    # print()
    # print('La ecuación del modelo es igual a:')
    # print('y = ', lr.coef_, 'x ', lr.intercept_)
    # print()
    # print("puntuacion aprendisaje final (estadística R al cuadrado) : ")
    # score = lr.score(X_train, y_train)
    # print(score)
    # y_pred=y_pred[:-1]
    # print("error cuadratico medio: ")
    # mse = mean_squared_error(y_train, y_pred)
    # print(mse)

    #guardamos los datos relevantes en imagen y archivo csv
    # modelo_reprobacion = [a, b, score, mse]
    # if score > 0.9:
    #     print("Guardo los datos")
    #     fig1.savefig("static/file/proyeccion_reprobacion.png")
    #     savetxt('Modelos/modelo_r_reprobacion.csv', modelo_reprobacion, delimiter=',')

    # plt.show()

    # #  ------------------------------------------------------------------------------------------

    print('DATOS DEL MODELO DE DESERCION')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    print('Coeficientes:',lr2.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_pred2 - y_test2) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % lr2.score(X_test2,y_test2))

    print('Regresión Lasso') 
    # Coeficiente
    print('Coeficientes:', rgl2.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_predrgl2 - y_test2) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % rgl2.score(X_test2, y_test2))

    print('Regresión Ridge')
    #Coeficiente
    print('Coeficientes:', rgr2.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_predrgr2 - y_test2) ** 2))
    # Varianza explicada
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

    # print('DATOS DEL MODELO DE DESERCION')
    # print()
    # print('Valor de la pendiente o coeficiente "a":')
    # a2 = lr2.coef_
    # print(a2)
    # print('Valor de la intersección o coeficiente "b":')
    # b2 = lr2.intercept_
    # print(b2)
    # print()
    # print('La ecuación del modelo es igual a:')
    # print('y = ', lr2.coef_, 'x ', lr2.intercept_)
    # print()
    # print("puntuacion aprendisaje final (estadística R al cuadrado) : ")
    # score2 = lr2.score(X_train2, y_train2)
    # print(score2)
    # y_pred2=y_pred2[:-1]
    # print("error cuadratico medio: ")
    # mse2 = mean_squared_error(y_train2, y_pred2)
    # print(mse2)

    # modelo_reprobacion2 = [a2, b2, score2, mse2]
    # if score2 > 0.8:
    #     print("Guardo los datos")
    #     fig2.savefig("static/file/proyeccion_desercion.png")
    #     savetxt('Modelos/modelo_r_desercion.csv', modelo_reprobacion2, delimiter=',')

    # plt.show()

    # #  ------------------------------------------------------------------------------------------

    print('DATOS DEL MODELO DE REPITENCIA')
    print ('Regresión Mínimos Cuadrados Ordinarios')
    # Coeficiente
    print('Coeficientes:',lr3.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_pred3 - y_test3) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % lr3.score(X_test3,y_test3))

    print('Regresión Lasso') 
    # Coeficiente
    print('Coeficientes:', rgl3.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_predrgl3 - y_test3) ** 2))
    # Varianza explicada
    print('Varianza explicada: %.2f\n' % rgl3.score(X_test3, y_test3))

    print('Regresión Ridge')
    #Coeficiente
    print('Coeficientes:', rgr3.coef_)
    # MSE
    print("Residual sum of squares: %.2f"% np.mean((y_predrgr3 - y_test3) ** 2))
    # Varianza explicada
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


    # print('DATOS DEL MODELO DE REPITENCIA')
    # print()
    # print('Valor de la pendiente o coeficiente "a":')
    # a3 = lr3.coef_
    # print(a3)
    # print('Valor de la intersección o coeficiente "b":')
    # b3 = lr3.intercept_
    # print(b3)
    # print()
    # print('La ecuación del modelo es igual a:')
    # print('y = ', lr3.coef_, 'x ', lr3.intercept_)
    # print()
    # print("puntuacion aprendisaje final (estadística R al cuadrado) : ")
    # score3 = lr3.score(X_train3, y_train3)
    # print(score3)
    # y_pred3=y_pred3[:-1]
    # print("error cuadratico medio: ")
    # mse3 = mean_squared_error(y_train3, y_pred3)
    # print(mse3)

    # modelo_reprobacion3 = [a3, b3, score3, mse3]
    # if score3 > 0.8:
    #     print("Guardo los datos")
    #     fig3.savefig("static/file/proyeccion_repitencia.png")
    #     savetxt('Modelos/modelo_r_repitencia.csv', modelo_reprobacion3, delimiter=',')

    # plt.show()





