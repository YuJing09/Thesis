import numpy as np

x=np.arange(10)
x=x.reshape([10,1])
vander=np.concatenate([x,x**2,x**3],axis=-1)
ip=np.linalg.pinv(vander)
ipp=(np.linalg.inv((vander.T).dot(vander))).dot(vander.T)
print(ip.dot(vander),ipp.dot(vander))
