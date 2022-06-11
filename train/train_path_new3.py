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
from modell import Ypgru
from modell2 import Ypdanet
from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4864)]
)
def gen2(hwm, host, port,m):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y= tup
    batch_size=X.shape[0]
    Y_t = Y[:, -1]
    X_tt= X[:,-1]
    rnn_state=np.zeros([batch_size,512])
    for i in range(X.shape[1]-1):
    
      
      #Y_t = Y[:, i]
        
        
      X_t = X[:, i]
   
      
      
      
      if i==0:
        
        rnn_state=np.zeros([batch_size,512])
        #if i != X.shape[1]-1:
        rnn_statee=m.predict(x=[X_t,rnn_state])
      else:
        
        rnn_state=rnn_statee[:,110:]
        #if i != X.shape[1]-1:
        
        rnn_statee=m.predict(x=[X_t,rnn_state])
      #print(rnn_state)
      #with tf.GradientTape() as tape:
     # predictions=m([X_t,rnn_state])
      #true_prob=Y_t[:,55]
      #pred_prob=tf.math.sigmoid(predictions[:,109])
      #print(true_prob,pred_prob)
      #loss=custom_loss(Y_t,predictions)
      #if loss>10000:
      #  print(Y_t,predictions,rnn_state)
      #var=0
      #for t in model.trainable_variables:
      #  if not tf.reduce_all(tf.equal(0,t)):
          #print(tf.reduce_prod(t.shape),t.shape)
      #    var+=batch_size*tf.reduce_prod(t.shape)
          
      #gradients= tape.gradient(loss, model.trainable_variables)
      #summ=[tf.reduce_sum(tf.abs(gradients[i])) for i in range(len(gradients))]
      #sum_bce_grad=tf.reduce_sum(summ)
      
      #print(sum_bce_grad/tf.cast(var,dtype=tf.float32))
    p=m.predict(x=[X_tt,rnn_state])
    loss=custom_loss(Y_t,p) 
    yield [X_tt,rnn_state], Y_t
def gen(hwm, host, port,m):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y= tup
    batch_size=X.shape[0]
    Y_t = Y[:, -1]
    X_tt= X[:,-1]
    #p=m.predict(x=X_tt)
    #loss=custom_loss(Y_t,p) 
    
    yield X_tt, Y_t
def custom_loss(y_true, y_pred,delta=2.):
  batch_size=tf.shape(y_true)[0]
  #valid_len=y_true[:,50]
  #valid_len=tf.reshape(valid_len,[-1,1])
  true_valid_len=tf.clip_by_value(y_true[:,50:51],0.,1000.)
  pred_valid_len=y_pred[:,50]
  true_std=tf.abs(y_true[:,:50]-y_pred[:,:50])
  pred_std=tf.math.exp(y_pred[:,51:101])
  
  #valid_len_loss=tf.reduce_sum(y_true[:,51]-y_pred[:,101])/tf.cast(batch_size,dtype=tf.float32)
  true_points=y_true[:,:50]
  pred_points=y_pred[:,:50]
  true_lead=y_true[:,51:55]
  
  true_lead_prob=y_true[:,55]
  pred_lead=y_pred[:,101:105]
  
  #print(true_lead,pred_lead)
  #true_lead_std=tf.abs(true_lead-pred_lead)
 
  #pred_lead_std=tf.math.exp(y_pred[:,105:109])
  #pred_lead_prob=tf.clip_by_value(tf.math.sigmoid(y_pred[:,105]),1e-15,1-1e-15)
  #lead_prob_mse=tf.square(pred_lead_prob-true_lead_prob)/2.
  #print(true_lead_prob,pred_lead_prob)  
  li=tf.cast(tf.clip_by_value(y_true[:,50]*10.,0.,50.),dtype=tf.float32)
 
 

  l=tf.cast(true_valid_len*10.,dtype=tf.int32)
  l=tf.clip_by_value(l,0,50)
  ll=tf.broadcast_to(l,(batch_size,50))
  len_x=tf.range(0,50,1,dtype=tf.int32)
  cond_len=tf.less(len_x,ll)
    #points huber loss 
  error_zero_point=tf.abs(pred_points[:,0])
  error_zero_point_std=tf.abs(tf.subtract(pred_points[:,0],pred_std[:,0]))
  error_point=tf.abs(tf.subtract(true_points[:,:],pred_points[:,:]))
  error_std=tf.abs(tf.subtract(true_std[:,:],pred_std[:,:]))
  error_valid_len=tf.abs(tf.subtract(true_valid_len[:,-1],pred_valid_len[:]))
  
  condition_zero_point=tf.less(error_zero_point,delta)
  condition_zero_point_std=tf.less(error_zero_point_std,delta)
  condition_point=tf.less(error_point,delta)
  condition_std=tf.less(error_std,delta)
  condition_valid_len=tf.less(error_valid_len,delta)
    
  small_res_point=0.5 * tf.square(error_point)  #mse
  small_res_std=0.5 * tf.square(error_std) #mse
  small_res_valid_len=0.5 * tf.square(error_valid_len)  #mse
  small_res_zero_point=0.5 * tf.square(error_zero_point) 
  small_res_zero_point_std=0.5 * tf.square(error_zero_point_std)
    
  large_res_point= delta * error_point - 0.5 * tf.square(delta) 
  large_res_std= delta * error_std - 0.5 * tf.square(delta) 
  large_res_valid_len=delta * error_valid_len - 0.5 * tf.square(delta)
  large_res_zero_point= delta * error_zero_point - 0.5 * tf.square(delta) 
  large_res_zero_point_std= delta * error_zero_point_std - 0.5 * tf.square(delta)
    
  point_loss=tf.where(condition_point,small_res_point,large_res_point)
  point_zero_point_loss=tf.where(condition_zero_point,small_res_zero_point,large_res_zero_point)
  point_zero_point_std_loss=tf.where(condition_zero_point_std,small_res_zero_point_std,large_res_zero_point_std)
  std_loss=tf.where(condition_std,small_res_std,large_res_std)
  valid_len_loss=tf.where(condition_valid_len,small_res_valid_len,large_res_valid_len)
  
  #point_loss=tf.reduce_sum(point_loss[:l])
 # std_loss=tf.reduce_sum(std_loss[:l])
  points_loss=tf.reduce_sum(tf.where(cond_len,point_loss,0),axis=1)
  stds_loss=tf.reduce_sum(tf.where(cond_len,std_loss,0),axis=1)
  
  #points loss
  points_all_loss=tf.where(tf.equal(li,0),point_zero_point_loss,(tf.math.divide(points_loss,li)))
  
  points_std_all_loss=tf.where(tf.equal(li,0),point_zero_point_std_loss,(tf.math.divide(stds_loss,li)))
  
  #lead huber loss
  error_lead=tf.abs(tf.subtract(true_lead[:,:],pred_lead[:,:]))
 # error_lead2=tf.abs(tf.subtract(true_lead[:],pred_lead[:]))
  
 
  #error_lead_std=tf.abs(tf.subtract(true_lead_std[:,:],pred_lead_std[:,:]))
    
  condition_lead=tf.less(error_lead,delta)
    
  #condition_lead_std=tf.less(error_lead_std,delta)
    
  small_res_lead=0.5 * tf.square(error_lead)
    
  #small_res_lead_std=0.5 * tf.square(error_lead_std) 
    
  large_res_lead= delta * error_lead - 0.5 * tf.square(delta) 
  
  #large_res_lead_std= delta * error_lead_std - 0.5 * tf.square(delta) 
    
  lead_loss=tf.where(condition_lead,small_res_lead,large_res_lead)
  #lead_std_loss=tf.where(condition_lead_std,small_res_lead_std,large_res_lead_std)
  
  #lead loss
  lead_all_loss=tf.where(tf.equal(true_lead_prob[:],1),tf.reduce_sum(lead_loss,axis=1),0)
  
  #lead_std_all_loss=tf.where(tf.equal(true_lead_prob[:],1),tf.reduce_sum(lead_std_loss,axis=1),0) 
  
    #bce
  true_prob=true_lead_prob[:]
  #pred_prob=pred_lead_prob[:]
  #bce=0.5 * tf.square(true_prob-pred_prob)
  #print(true_prob-pred_prob)
  #error_prob=tf.abs(tf.subtract(pred_lead_prob,true_lead_prob))
  #bce=tf.where(tf.less(error_prob,0.8),bce,0.)
  #print(points_all_loss,lead_all_loss,valid_len_loss)
    #bce=-1.*(true_prob*tf.math.log(pred_prob)+((1.-true_prob)*tf.math.log(1.-pred_prob)))
    #s add all loss
  #print(points_all_loss,points_std_all_loss,bce,lead_all_loss,lead_std_all_loss,valid_len_loss)
  
  #loss=points_all_loss+points_std_all_loss/1000.+lead_all_loss+lead_std_all_loss/1000.+valid_len_loss+bce
  loss=points_std_all_loss/1000.+points_all_loss+lead_all_loss+valid_len_loss
    #all_loss=points_all_loss
  #print(loss)  
    #
    #print(points_all_loss,points_std_all_loss,lead_all_loss,lead_std_all_loss,valid_len_loss)
    
  

  
  return loss
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=30, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')

  args = parser.parse_args()
 
  model=Ypdanet(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1)
  model =tf.keras.Model(inputs=[input1],outputs=output)
  
  adam=optimizers.Adam(lr=0.0001)
  filepath="./saved_model/weights3-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
  callbacks_list = [checkpoint]
 
  
    #print(layer.name,layer.trainable)
  
  #print(x.shape,y.shape)
  model.summary()
  
  model.compile(optimizer=adam, loss=custom_loss)
  model.load_weights('./saved_model/weights3-improvement-30-2.06.hdf5',by_name=True)
 
  history=model.fit_generator(
    gen(20, args.host, port=args.port,m=model),
    steps_per_epoch=50*1200/32.,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port,m=model),
    validation_steps=25*1200/32.,verbose=1,callbacks=callbacks_list)

  np.save('./loss_history/loss_danet6',np.array(history.history['loss']))
  np.save('./loss_history/val_loss_danet6',np.array(history.history['val_loss']))
