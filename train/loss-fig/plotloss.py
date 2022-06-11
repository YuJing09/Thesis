import numpy as np
import  matplotlib.pyplot as plt
a=np.linspace(0+1e-15,1-1e-15,500)

def bce(a,label=1):
  loss=-1*(label*np.log(a)+(1-label)*np.log(1-a))
  return loss
bceloss=bce(a)
plt.plot(a[10:],bceloss[10:])
plt.show()
