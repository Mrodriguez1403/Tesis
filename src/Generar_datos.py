import pandas as pd
import numpy as np
from numpy import savetxt, loadtxt
import random

datos = pd.read_csv('Datos/Datos_MEN.csv', header=0)

y = datos['REPROBACIÓN_MEDIA'].values
y2 = datos['DESERCIÓN_MEDIA'].values
y3 = datos['REPITENCIA_MEDIA'].values

datos_reprobacion = []
datos_desercion = []
datos_repitencia = []

may = max(y)
may2 = max(y2)
may3 = max(y3)

men = min(y)
men2 = min(y2)
men3 = min(y3)

def generar_datos(menor, mayor,z):
    d = []
    for i in range(0, 41):
        r = random.uniform(menor, mayor)
        r = round(r, 4)
        d.append(r)
        
    for i in z:
        d.append(i)
    return d

datos_reprobacion = generar_datos(men, may,y)
datos_desercion = generar_datos(men2, may2,y2)
datos_repitencia = generar_datos(men3, may3,y3)

periodos = []

for i in range(1970, 2020):
    periodos.append(i)


print("--- PERIODOS ---")
print(periodos)
print()
print("--- REPROBACION ---")
print(datos_reprobacion)
print()
print("--- DERSERCION ---")
print(datos_desercion)
print()
print("--- REPITENCIA ---")
print(datos_repitencia)


savetxt('Datos/datos_Reprobacion.csv', datos_reprobacion, delimiter=',')
savetxt('Datos/datos_Desercion.csv', datos_desercion, delimiter=',')
savetxt('Datos/datos_Repitencia.csv', datos_repitencia, delimiter=',')

savetxt('Datos/Periodos.csv', periodos, delimiter=',')

# d = loadtxt('datos_hibridos.csv',delimiter=',')
