import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from numpy import savetxt, loadtxt
# -------------------------------------------------------------------------------

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

# Desercion media
lr2= LinearRegression()

# Repitencia media
lr3= LinearRegression()

# #entreno el modelo  ----------------------------------------------------------------------------------

# Reprobacion media
lr.fit(X_train, y_train)

# Desercion media
lr2.fit(X_train2, y_train2)

# Repitencia media
lr3.fit(X_train3, y_train3)



#realizamos la prediccion  ---------------------------------------------------------------------------------

# Reprobacion media
y_pred = lr.predict(X_test)

# Desercion media
y_pred2 = lr.predict(X_test2)

# Repitencia media
y_pred3 = lr.predict(X_test3)


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
ax2.plot(X_test,y_pred,color='red')
ax2.set_title('Regresion Lineal')
ax2.set_xlabel('Periodos')
ax2.set_ylabel('Prediccion Reprobacion Media')

print('DATOS DEL MODELO DE REPROBACION')
print()
print('Valor de la pendiente o coeficiente "a":')
a = lr.coef_
print(a)
print('Valor de la intersección o coeficiente "b":')
b = lr.intercept_
print(b)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr.coef_, 'x ', lr.intercept_)
print()
print("puntuacion aprendisaje final (estadística R al cuadrado) : ")
score = lr.score(X_train, y_train)
print(score)
print("error cuadratico medio: ")
mse = mean_squared_error(y_train, y_pred)
print(mse)

modelo_reprobacion = [a, b, score, mse]
if score > 0.1:
    fig1.savefig("static/file/p_s_reprobacion.png")
    savetxt('Modelos/modelo_s_reprobacion.csv', modelo_reprobacion, delimiter=',')

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
ax2.plot(X_test2,y_pred2,color='red')
ax2.set_title('Regresion Lineal')
ax2.set_xlabel('Periodos')
ax2.set_ylabel('Prediccion Desercion Media')

print('DATOS DEL MODELO DE DESERCION')
print()
print('Valor de la pendiente o coeficiente "a":')
a2 = lr2.coef_
print(a2)
print('Valor de la intersección o coeficiente "b":')
b2 = lr2.intercept_
print(b2)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr2.coef_, 'x ', lr2.intercept_)
print()
print("puntuacion aprendisaje final (estadística R al cuadrado) : ")
score2 = lr2.score(X_train2, y_train2)
print(score2)
print("error cuadratico medio: ")
mse2 = mean_squared_error(y_train2, y_pred2)
print(mse2)

modelo_reprobacion2 = [a2, b2, score2, mse2]
if score2 > 0.1:
    fig2.savefig("static/file/p_s_desercion.png")
    savetxt('Modelos/modelo_s_desercion.csv', modelo_reprobacion2, delimiter=',')

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
ax2.plot(X_test3,y_pred3,color='red')
ax2.set_title('Regresion Lineal')
ax2.set_xlabel('Periodos')
ax2.set_ylabel('Prediccion Repitencia Media')

print('DATOS DEL MODELO DE REPITENCIA')
print()
print('Valor de la pendiente o coeficiente "a":')
a3 = lr3.coef_
print(a3)
print('Valor de la intersección o coeficiente "b":')
b3 = lr3.intercept_
print(b3)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr3.coef_, 'x ', lr3.intercept_)
print()
print("puntuacion aprendisaje final (estadística R al cuadrado) : ")
score3 = lr3.score(X_train3, y_train3)
print(score3)
print("error cuadratico medio: ")
mse3 = mean_squared_error(y_train3, y_pred3)
print(mse)

modelo_reprobacion3 = [a3, b3, score3, mse3]
if score3 > 0.1:
    fig3.savefig("static/file/p_s_repitencia.png")
    savetxt('Modelos/modelo_s_repitencia.csv', modelo_reprobacion3, delimiter=',')

# plt.show()




