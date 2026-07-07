import numpy as np


# DATOS DEL PROBLEMA
r = 0.1
g = 32.1
C = 0.5*(np.pow(r, 2))*np.sqrt(2*g)


# PARAMETROS DE INICIALIZACION
h = 20
n = 80
x0 = 0.0    # Valor del tiempo
y0 = np.array([[8.0]], dtype = float) #valor de X = h (altura del nivel del agua)


################RUTINA023-Runge-kutta (Rk4)##########################
m=len(y0)
a=np.hstack((np.zeros((m-1,1)),np.eye((m-1))))
A=lambda x:np.vstack((a,np.array(([0]),dtype=float)))
B=lambda x, y:np.vstack((np.zeros((m-1,1)),np.array(([-C*np.pow(y, -1.5)]),dtype=float)))
x=x0
for i in range(n):
    dy0=A(x)@y0+B(x, y0)
    x+=h/2
    y1=y0+(h/2)*dy0
    dy1=A(x)@y1+B(x, y1)
    y2=y0+(h/2)*dy1
    dy2=A(x)@y2+B(x, y2)
    x+=h/2
    y3=y0+h*dy2
    dy3=A(x)@y3+B(x, y3)
    y=y0+(h/6)*(dy0+2*dy1+2*dy2+dy3)
    y0=y.copy()


    print(f" Altura H2O= {y[0]}, TIEMPO= {x}, Iteraciones= {i+1}, RAPID= {dy1} ")

# 1)  EL NIVEL DEL LIQUIDO A LOS 60 SEG
print("El nivel del liquido a los 60 segundos es: ", 7.87096533)


#2)   VOLUMEN DEL LIQUIDO A LOS 80 SEGUNDOS
volumen = 1/3*(np.pi*7.8272415*np.pow(7.8272415, 2) ) # 1/3*pi*r2*h

print("El volumen del liquido a los 80 seg es: ", volumen)


#3)   EL AREA DE LA SUPERFICIE LIBRE A LOS 100 SEG
area = np.pi*np.pow(7.78314819, 2)
print("El area de la superficie libre a los 100 seg es: ", area)


#4)   EL TIEMPO EN MIN EN NQUE SE DESCARGA EL TANQUE 
print("El tiempo en que se descarga todo el tanque es: ", 1500/60)


#5)   LA RAPIDEZ CON LA QUE DISMINUYE LA SUPERFICIE DEL LIQUIDO A LOS 20 SEG
print("La rapidez del liquido a los 20 seg es: ", -0.00213312)