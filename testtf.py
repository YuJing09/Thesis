import tensorflow as tf 
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = " "
batch_size=1

label=tf.random.uniform((batch_size,50),minval=-1,maxval=1,dtype=tf.float32)
vaild_len=tf.random.uniform((batch_size,1),minval=0,maxval=70,dtype=tf.float32)
stds=tf.random.uniform((batch_size,50),minval=0,maxval=1,dtype=tf.float32)
ground_truth=tf.random.uniform((batch_size,50),minval=-1,maxval=1,dtype=tf.float32)
ground_len=tf.random.uniform((batch_size,1),minval=0,maxval=70,dtype=tf.float32)
ss=tf.shape(label)
l=tf.cast(100,dtype=tf.int32)
#print(label[0,:l])
#print(label)
#sts=tf.random.uniform((batch_size,5),minval=0,maxval=1,dtype=tf.float32)
#stds=tf.random.uniform((batch_size,1,1,10),minval=0,maxval=1,dtype=tf.float32)
#stds2=tf.random.uniform((batch_size,1,1,1),minval=0,maxval=1,dtype=tf.float32)
#print(stds,stds2,stds2-stds)

#print(vaild_len)
#print(stds)
label=tf.concat([label,stds,vaild_len],axis=-1)

#print(vaild_l)
#bb=tf.range(1,a,1,dtype=tf.float32)
#b=tf.reshape(a,[-1,1])
                                                                     #ground_truth: 51  50 y-points + 1  x-vaild_len   mse: velocity 1
                                                                     #predict : 50 y-points 50 y stds 1  x-vaild_len  mse: velocity 1
l=tf.cast(vaild_len,dtype=tf.int32)
l=tf.clip_by_value(l,0,50)
ll=tf.broadcast_to(l,(batch_size,50))
#print(tf.reduce_sum(ll,axis=1))
len_x=tf.range(0,50,1,dtype=tf.int32)

cond_len=tf.less(len_x,ll)
                                                                       
a=tf.cast(np.log(1-1e-15),dtype=tf.float32)
#print(a)
points=label[:,:50]
#print(points/tf.cast(l,dtype=tf.float32))
allmse=tf.square(points-ground_truth)
a=tf.abs(points[:,:50]-ground_truth[:,:50])
#print(points[:,30:31])

#print(tf.reduce_all(tf.less(label,100)))
vaild_len_loss=tf.reduce_sum(tf.square(vaild_len-ground_len))/tf.cast(batch_size,dtype=tf.float32)
#allmse=tf.where(cond,allmse,0)
#print(allmse)
s=0.
for i in range(batch_size):
  
  l=tf.cast(ground_len[i][0],dtype=tf.int32)
  mse=allmse[i,:l]
  l=tf.clip_by_value(l,3,50)
  
  vandermonde=tf.range(0,50,1,dtype=tf.float32)
 
  vandermonde2=tf.square(vandermonde)
  vandermonde3=tf.math.pow(vandermonde,3)
  v1=tf.reshape(vandermonde,[-1,1])
  v2=tf.reshape(vandermonde2,[-1,1])
  v3=tf.reshape(vandermonde3,[-1,1])
  
  vandermonde=tf.concat([v1,v2,v3],axis=-1)
  #print(vandermonde)
  
  points=label[i,:50]
  points=points-points[0]
  std=stds[i,:]
  std=tf.reshape(std,(-1,1))
  points=tf.reshape(points,(-1,1))
  y=points*std
  x=vandermonde[:]*std
 
  vander=tf.linalg.pinv(x)
  p=tf.matmul(vander,y)
  #print(p)
  #eee=np.linalg.inv((x.T).dot(X)).dot(x.T)
  #print(vandermonde,eee)
  y_pred=tf.matmul(vandermonde,p)[:,-1]
  #print(y_pred)
  y_true=ground_truth[i,:]
  
  mse_std=tf.square(y_pred-y_true)
  
  point_loss=tf.where(tf.less(l,4),tf.reduce_sum(mse),tf.reduce_sum(mse_std[:l]))
  
  s=s+point_loss
all_loss=s/tf.cast(batch_size,dtype=tf.float32)+vaild_len_loss
print(all_loss/1)
