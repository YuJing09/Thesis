import tensorflow as tf
import argparse
import json 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from modell_transform import Ypgru
from modell2 import Ypdanet
from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
#from train_pathmodel import gen,custom_loss
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
    print(y_p[:l],y_p_v[:l],y_t[:l],weight)
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
    bce=tf.where(tf.less(error_prob,0.8),bce,0.)
    s=s+loss_v+bce+valid_len_loss+loss_p+lead_all_loss
    
  all_loss=s/tf.cast(batch_size,dtype=tf.float32)
  
  return all_loss

import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = " "
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--model_weights', type=str, default="weights-danet-improvement-04-3.05.hdf5", help='Data server ip address.')
    args = parser.parse_args()
    model=Ypdanet(2)
    input1=tf.keras.Input(shape=(160,320,3),name='img')
  
    input2=tf.keras.Input(shape=(512),name='rnn')
    output=model.call(input1)
    model =tf.keras.Model(inputs=[input1],outputs=output)
    weights='./saved_model/'+args.model_weights
    model.load_weights(weights)
    #testdatapath='./14/camera320.h5'
    testdatapath='../Chunk_3/99c94dc769b5d96e_2018-05-13--12-07-39/27/camera320.h5'
    pathdata=testdatapath.replace('camera320','pathdata')
    radarpath=testdatapath.replace('camera320','radardata')
    with h5py.File(pathdata,'r') as t5:
      c5=h5py.File(testdatapath,'r')
      img=c5['X']
      l5=h5py.File(radarpath,'r')
      Path=t5['Path'][:]
      leadOne=l5['LeadOne'][:]
      out=np.concatenate((Path,leadOne),axis=-1)
      out[:,50:52]/=10.
      rnn_state=np.zeros([1,512])
      loss_all=0
      #i=500
      for i in range(len(out)):
        prediction=model.predict(x=[img[i:i+1]])
        #prediction[:,100:102]/=1.
        #prediction[:,105]/=100.
        #rnn_state=prediction[:,106:]
        loss=custom_loss(out[i:i+1],prediction)
        #loss2=custom_loss0(out[i:i+3],prediction)
        #loss_all+=loss
        loss_all+=loss
        print(loss)
     # i ==500:
      #print(point,std,lenn,lead,lead_std,prob,out[i],loss)
      print(loss_all/len(out))
     
