import matplotlib.pyplot as plt
# importamos la libreria de matplotlib para realizar graficas
import numpy as np
# importamos la libreria de numpy para realizar calculos estadisticos
import pandas as pd
# importamos libreria pandas para leer y utilizar datos csv

# datos=pd.read_csv('Prueba.csv',header=0)
def generar_graficas_reales():
    datos=pd.read_csv('Datos/Datos_MEN.csv',header=0)
    # importamos las datos csv con encabezado y lo guardamos en una variable
    datos['AÑO']=datos['AÑO'].astype('int64')
    x=datos['AÑO'].values

    # guardamos la columna AÑO en una varible

    y=datos['REPROBACIÓN_MEDIA'].values
    y2=datos['DESERCIÓN_MEDIA'].values
    y3=datos['REPITENCIA_MEDIA'].values


    # guardamos la columna reprobacion_media en una varible

    fig1 = plt.figure(figsize=(18,12), dpi=100)

    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig1.add_subplot(2, 2, 1)
    ax.plot(x,y,color="blue")
    ax.set_xlabel('periodos') 
    ax.set_ylabel('tasa de reprobacion')
    ax.set_title('grafica de reprobacion')

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.plot(x,y2,color="red")
    ax2.set_xlabel('periodos') 
    ax2.set_ylabel('tasa de deserción')
    ax2.set_title('grafica de deserción')

    ax3= fig1.add_subplot(2, 2, 3)
    ax3.plot(x,y3,color="yellow")
    ax3.set_xlabel('periodos') 
    ax3.set_ylabel('tasa de repetición')
    ax3.set_title('grafica de repitencia')

    ax4= fig1.add_subplot(2, 2, 4)
    ax4.plot(x,y,color="blue",label='reprobacion')
    ax4.plot(x,y2,color="red",label='desercion')
    ax4.plot(x,y3,color="yellow",label='repitencian')
    ax4.set_xlabel('periodos') 
    ax4.set_ylabel('tasas')
    ax4.set_title('conjunto de graficas')

    ruta = "static/file/grafica_real.png"
    fig1.savefig(ruta)
    return ruta
    # plt.show()
