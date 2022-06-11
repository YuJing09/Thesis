import numpy as np
import matplotlib.pyplot as plt
def rot(theta):
  return np.array([[np.cos(theta),np.sin(theta)],[-1*np.sin(theta),np.cos(theta)]])
theta=np.pi/4	
x=np.array([1,1])
rota=rot(theta)

xx=rota.dot(x)
print(x,xx)
a=np.arange(9)
a=a.reshape([3,3])
b=np.ones([3,3])
c=np.einsum('ij,kj->ki',a,b)
print(a.dot(b),c)
