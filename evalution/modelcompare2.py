import tensorflow as tf
import argparse
import json 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from modell_transform import Ypgru,Ypgru2
from modell2 import Ypdanet,Ypreg
from server import client_generator
from server_path import path
import math
os.environ['CUDA_VISIBLE_DEVICES'] = " "
def vandermondepoints(predict,l):
  vandermonde=tf.range(0,50,1,dtype=tf.float32)
  
  p=predict[:,:50]
  #l_v=np.clip(((predict[:,50]*10.).astype(np.int32)),0,50)
  w=predict[:,51:101]
  vandermonde2=tf.square(vandermonde)
  vandermonde3=tf.math.pow(vandermonde,3)
  v1=tf.reshape(vandermonde,[-1,1])
  v2=tf.reshape(vandermonde2,[-1,1])
  v3=tf.reshape(vandermonde3,[-1,1])
  p_vs=[]
  vandermonde=tf.concat([v1,v2,v3],axis=-1)
  batch_size=tf.shape(p)[0]
  w=tf.math.exp(w)
  
  for i in range(batch_size):
    #l=int(l_v[i])
    
    point=p[i,:l]
    point=point-point[0]
    weight=w[i,:l]**0.1
    weight=tf.reshape(weight,(-1,1))
    #print(weight)
    point=tf.reshape(point,(-1,1))
    y=point/weight
    
    x=vandermonde[:l]/weight
    #print(x,y)
    vander=tf.linalg.pinv(x)
    
    pv=tf.matmul(vander,y)
    
    y_p_v=tf.matmul(vandermonde[:l],pv)[:,-1]+p[i,0]
    #print(y_p_v,p)
    p_vs.append(y_p_v)
  return p_vs
def custom_loss(y_true, y_pred,delta=2.):
  batch_size=tf.shape(y_true)[0]
  valid_len=y_true[:,50]*10.
  valid_len=tf.reshape(valid_len,[-1,1])
  true_valid_len=y_true[:,50]
  pred_valid_len=y_pred[:,50]
  
  vandermonde=tf.range(0,50,1,dtype=tf.float32)
 
  vandermonde2=tf.square(vandermonde)
  vandermonde3=tf.math.pow(vandermonde,3)
  v1=tf.reshape(vandermonde,[-1,1])
  v2=tf.reshape(vandermonde2,[-1,1])
  v3=tf.reshape(vandermonde3,[-1,1])
  
  vandermonde=tf.concat([v1,v2,v3],axis=-1)
  weights=tf.nn.softmax(y_pred[:,51:101],axis=-1)
  #valid_len_loss=tf.reduce_sum(y_true[:,51]-y_pred[:,101])/tf.cast(batch_size,dtype=tf.float32)
  true_points=y_true[:,:50]
  pred_points=y_pred[:,:50]
  true_lead=y_true[:,51:55]
  pred_lead=y_pred[:,101:105]
  true_lead_prob=y_true[:,55]
  pred_lead_prob=tf.clip_by_value(tf.math.sigmoid(y_pred[:,105]),1e-15,1-1e-15)
  s=0.
  loss=tf.zeros([4],tf.float32)
  for i in range(batch_size):
    
    l=tf.cast(valid_len[i][0],dtype=tf.int32)
   
    l=tf.clip_by_value(l,0,50)
    error_lead=tf.abs(tf.subtract(true_lead[i,:],pred_lead[i,:]))
   
    
    condition_lead=tf.less(error_lead,delta)
    

    
    small_res_lead=0.5 * tf.square(error_lead)
    
   
    
    large_res_lead= delta * error_lead - 0.5 * tf.square(delta) 
  
    
    
    lead_loss=tf.where(condition_lead,small_res_lead,large_res_lead)
   
    
    #lead loss
    lead_all_loss=tf.where(tf.equal(true_lead_prob[i],1),tf.reduce_sum(lead_loss),0)
    
    
    point=pred_points[i,:]
    point=point-point[0]
    weight=weights[i,:]
    weight=tf.reshape(weight,(-1,1))
    point=tf.reshape(point,(-1,1))
    y=point*weight
    #y=y[:l]
    
    x=vandermonde[:]*weight
    #x=x[:l]
    vander=tf.linalg.pinv(x)
    p=tf.matmul(vander,y)
    y_p = y_pred[i,:50]
    y_p_v=tf.matmul(vandermonde,p)[:,-1]+pred_points[i,0]
    y_t=y_true[i,:50]
    #print(y_p[:l],y_p_v[:l],y_t[:l],weight)
    #huber loss 
 
    error_points= tf.abs(tf.subtract(y_p,y_t))
    error_points_v= tf.abs(tf.subtract(y_p_v,y_t))
    
    condition_points=tf.less(error_points,delta)
    condition_points_v=tf.less(error_points_v,delta)
    
    small_res_points=0.5 * tf.square(error_points)  #mse
    small_res_points_v=0.5 * tf.square(error_points_v)  #mse
    
    large_res_points= delta * error_points - 0.5 * tf.square(delta)
    large_res_points_v= delta * error_points_v - 0.5 * tf.square(delta)
    points_loss=tf.where(condition_points,small_res_points,large_res_points)
    
    points_v_loss=tf.where(condition_points_v,small_res_points_v,large_res_points_v)  
    
    
    loss_v=tf.reduce_sum(points_v_loss[:l])
    loss_p=tf.reduce_sum(points_loss[:l])

    error_valid_len=tf.abs(tf.subtract(true_valid_len[i],pred_valid_len[i]))
    condition_valid_len=tf.less(error_valid_len,delta)
    small_res_valid_len=0.5 * tf.square(error_valid_len)  #mse
    large_res_valid_len=delta * error_valid_len - 0.5 * tf.square(delta)
    valid_len_loss=tf.where(condition_valid_len,small_res_valid_len,large_res_valid_len)
    true_prob=true_lead_prob[i]
    pred_prob=pred_lead_prob[i]
    bce=0.5 * tf.square(true_prob-pred_prob)
    
    error_prob=tf.abs(tf.subtract(pred_prob,true_prob))
    #print(true_prob,pred_prob)
    #bce=tf.where(tf.less(error_prob,0.8),bce,0.)
    s=s+loss_v+bce+valid_len_loss+loss_p+lead_all_loss
    aa=tf.cast([loss_p,valid_len_loss,lead_all_loss,bce],dtype=tf.float32)
    loss+=aa
    
    
  all_loss=s/tf.cast(batch_size,dtype=tf.float32)
  loss=loss/tf.cast(batch_size,dtype=tf.float32)
  return all_loss,loss
def RMSE(x,y):
  y=np.array(y)
  
  
  return (np.sum(0.5*(x-y)**2))


if __name__=="__main__":
  model=Ypdanet(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1)
  model =tf.keras.Model(inputs=[input1],outputs=output)
  model1=Ypreg(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  model2=Ypgru(2)
  output2=model2.call(input1,input2)
  model2=tf.keras.Model(inputs=[input1,input2],outputs=output2)
  #input2=tf.keras.Input(shape=(512),name='rnn')
  output=model1.call(input1)
  model1 =tf.keras.Model(inputs=[input1],outputs=output)
  #model3=Ypgru2(2)
  #output3=model3.call(input1,input2)
  #model3=tf.keras.Model(inputs=[input1,input2],outputs=output3)
  #model.load_weights('./saved_model/weights3-improvement-01-0.42.hdf5')
  #model1.load_weights('./saved_model/weights-reg-improvement-20-2.79-final.hdf5')
  #model2.load_weights('./saved_model/weights-v-improvement-01-2.84.hdf5')
  #model3.load_weights('./saved_model/weights-v-improvement-01-2.96.hdf5')
  data_len=len(path)
  test_path=path[-1*int(0.1*data_len):]
  loss=[]
  vandermonde=np.reshape(np.arange(50),[-1,1])
  vandermonde=np.concatenate([vandermonde,vandermonde**2,vandermonde**3],axis=-1)
  loss_l=4
  cc=np.zeros([loss_l],dtype=np.float32)
  dd=np.zeros([loss_l],dtype=np.float32)
  ee=np.zeros([loss_l],dtype=np.float32)
  ff=np.zeros([1],dtype=np.float32)
  gg=np.zeros([1],dtype=np.float32)
  r=0
  q=np.zeros([1])
  rad=0
  for i in test_path:
    with h5py.File(i,'r') as c:
      try:
        path_data=i.replace('camera320.h5','pathdata.h5')
        radarpath=i.replace('camera320','radardata')
      

        path=h5py.File(path_data,'r')
        l5=h5py.File(radarpath,'r')
        P=path['Path']
        img=c['X']
      except:
        continue
      leadOne=l5['LeadOne'][:]
      rnn_state0=np.zeros([1,512])
      rnn_state1=np.zeros([1,512])
      rnn_state=np.zeros([1,512])
      l1=[]
      l2=[]
      l3=[]
      l4=[]
      a=np.zeros([loss_l],dtype=np.float32)
      b=np.zeros([loss_l],dtype=np.float32)
      d=np.zeros([loss_l],dtype=np.float32)
      loss_v=[]
      loss_p=[]
      #e=np.zeros([5],dtype=np.float32)
      out=np.concatenate((P,leadOne),axis=-1)
      out[:,50:52]/=10.
      r+=1
      q+=math.ceil(len(img)/5)
      rad+=np.sum(out[:,-1][range(0,len(img),5)])
  print(q,rad)

        
