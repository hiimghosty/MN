import numpy as np
np.set_printoptions(suppress=True)
x0=8 #radio inicial, se despeja
g=32.1 #pies
y0 = np.array(([8]),dtype=float) #condiciones iniciales
def areax(x):
    return np.pi*x**2
################RUTINA023########################## #RK04
m=len(y0)
h=20
n=100 #número de pasos
resultados=np.zeros((n+1,2+m))
resultados[0, :] = np.concatenate(([0, x0], y0.flatten()))
a=np.hstack((np.zeros((m-1,1)),np.eye((m-1))))
A=lambda x:np.vstack((a,np.array(([0]),dtype=float)))
B=lambda x:np.vstack((np.zeros((m-1,1)),np.array(([-0.5*np.pi*(x**2)*(np.sqrt(2*g) * ((np.sqrt(x))/(areax(x)))
)]),dtype=float)))
x=x0
for i in range(n):
    dy0=A(x)@y0+B(x)
    x+=h/2
    y1=y0+(h/2)*dy0
    dy1=A(x)@y1+B(x)

    y2=y0+(h/2)*dy1
    dy2=A(x)@y2+B(x)

    x+=h/2
    y3=y0+h*dy2
    dy3=A(x)@y3+B(x)

    y=y0+(h/6)*(dy0+2*dy1+2*dy2+dy3)
    y0=y.copy()
    resultados[i+1, :] = np.concatenate(([i+1,x], y0.flatten()))

print(resultados)