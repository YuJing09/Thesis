import numpy as np
import matplotlib.pyplot as plt
x=np.arange(10).reshape([10,1])
w=np.arange(10).reshape([10,1])+1
vander=np.concatenate([x,x**2,x**3],axis=-1)
y=np.array([1,6,7,8,11,12,15,18,22,30])
y=y.reshape([10,1])-1

I=np.eye(10)
#print(u,d,vt)
#print(u1,d1,vt1)
II=np.eye(10)
for i in range(10):
  try:
    I[i][i]=1/w[i]**3
  except:
    I[i][i]=0
#print(I.dot(vander))
print(I)
ip=np.linalg.pinv(I.dot(vander))
ipp=np.linalg.pinv(vander)

p=ip.dot(I.dot(y))
pp=ipp.dot(y)
y_=vander.dot(p)+1
yy_=vander.dot(pp)+1
plt.scatter(x[:,0],y[:,0]+1)
plt.plot(x[:,0],y_[:,0],color='r')
plt.plot(x[:,0],yy_[:,0],color='b')

plt.savefig('imgg.png')


