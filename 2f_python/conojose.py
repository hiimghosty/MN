import numpy as np 

#x'=-0.5*np.pi*r**2*np.sqrt(2*g)*(np.sqrt(x)/np.pi*x)

r=0.1 
g=32.1
y0=np.array([[8.0]],dtype=float)
x0=0.0
V0=(512/3)*np.pi
h=20
n=77
C=-0.5*np.pow(r,2)*np.pow(2*g,0.5)
m=len(y0)
a=np.hstack((np.zeros((m-1,1)),np.eye((m-1))))
A=lambda x,y:np.vstack((a,np.array(([C*(np.pow(y,0.5)/np.pow(y,3))]),dtype=float)))
B=lambda x:np.vstack((np.zeros((m-1,1)),np.array(([0]),dtype=float)))
x=x0
for i in range(n):
    dy0=A(x,y0[0])@y0+B(x)
    x+=h/2
    y1=y0+(h/2)*dy0
    dy1=A(x,y1[0])@y1+B(x)
    y2=y0+(h/2)*dy1
    dy2=A(x,y2[0])@y2+B(x)
    x+=h/2
    y3=y0+h*dy2
    dy3=A(x,y3[0])@y3+B(x)
    y=y0+(h/6)*(dy0+2*dy1+2*dy2+dy3)
    y0=y.copy()
    print("En t=",x,"[seg]","La altura es: ",y[0])