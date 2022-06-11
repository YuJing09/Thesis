import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
batch_size=2
points=np.random.randint(0,5,[batch_size,50])
valid_len=np.random.randint(0,80,[batch_size,1])
dRel=np.random.randint(0,100,[batch_size,1])
yRel=np.random.randint(0,5,[batch_size,1])
vRel=np.random.randint(0,20,[batch_size,1])
aRel=np.random.randint(0,5,[batch_size,1])
x_train=np.concatenate([points,valid_len,dRel,yRel,vRel,aRel],axis=-1)
class Normalization():
  def __init__(self,x_train,x_validation=None,x_test=None):
    self.train=x_train
    self.validation=x_validation
    self.test=x_test
    self.standardscalers=[StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler()]
    self.normalize(self.train)
  def normalize(self,x_train):
    self.valid_index=np.clip(x_train[:,50],0,50)
    self.points=np.concatenate([x_train[i,:int(self.valid_index[i])] for i in range(len(self.train))],axis=-1)
    print(self.points)
    self.standardscalers[0].fit(self.points)
    #self.standardscalers[1].fit(self.train[:,50])
    #self.standardscalers[2].fit(self.train[:,51])
    #self.standardscalers[3].fit(self.train[:,52])
    #self.standardscalers[4].fit(self.train[:,53])
    #self.standardscalers[5].fit(self.train[:,54])
    #print(self.standardscalers[1].transform(x_train[:,50]))

s=StandardScaler()
s.fit(points)
std_p=s.transform(points)
print(len(s.var_))
