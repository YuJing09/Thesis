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
  valid_len=y_true[:,50]*10.
  valid_len=tf.reshape(valid_len,[-1,1])
  true_valid_len=y_true[:,50]
  pred_valid_len=y_pred[:,50]
  
  
  
  
  pred_abs=tf.math.exp(y_pred[:,51:101])
  #valid_len_loss=tf.reduce_sum(y_true[:,51]-y_pred[:,101])/tf.cast(batch_size,dtype=tf.float32)
  true_points=y_true[:,:50]
  pred_points=y_pred[:,:50]
  true_abs=tf.abs(tf.subtract(true_points,pred_points))
  true_lead=y_true[:,51:55]
  pred_lead=y_pred[:,101:105]
  true_lead_prob=y_true[:,55]
  
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
    
    
   
    y_p = y_pred[i,:50]
    
    y_t=y_true[i,:50]
    er_t=true_abs[i,:50]
    er_p=pred_abs[i,:50]
    #huber loss 
 
    error_points= tf.abs(tf.subtract(y_p,y_t))
    error_points_e= tf.abs(tf.subtract(er_t,er_p))
    
    condition_points=tf.less(error_points,delta)
    condition_points_e=tf.less(error_points_e,delta)
    
    small_res_points=0.5 * tf.square(error_points)  #mse
    small_res_points_e=0.5 * tf.square(error_points_e)  #mse
    
    large_res_points= delta * error_points - 0.5 * tf.square(delta)
    large_res_points_e= delta * error_points_e - 0.5 * tf.square(delta)
    points_loss=tf.where(condition_points,small_res_points,large_res_points)
    points_e_loss=tf.where(condition_points_e,small_res_points_e,large_res_points_e)  
    
    
    loss_e=tf.reduce_sum(points_e_loss[:l])/tf.cast(tf.clip_by_value(l,1,50),dtype=tf.float32)
    loss_p=tf.reduce_sum(points_loss[:l])/tf.cast(tf.clip_by_value(l,1,50),dtype=tf.float32)
    error_valid_len=tf.abs(tf.subtract(true_valid_len[i],pred_valid_len[i]))
    condition_valid_len=tf.less(error_valid_len,delta)
    small_res_valid_len=0.5 * tf.square(error_valid_len)  #mse
    large_res_valid_len=delta * error_valid_len - 0.5 * tf.square(delta)
    valid_len_loss=tf.where(condition_valid_len,small_res_valid_len,large_res_valid_len)
    
    
    
    s=s+loss_e/1000.+valid_len_loss+loss_p+lead_all_loss/10.
    
  all_loss=s/tf.cast(batch_size,dtype=tf.float32)
  
  return all_loss
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
  filepath="./saved_model/weights4-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
  callbacks_list = [checkpoint]
 
  
    #print(layer.name,layer.trainable)
  
  #print(x.shape,y.shape)
  #model.summary()
  
  model.compile(optimizer=adam, loss=custom_loss)
  #model.load_weights('./saved_model/weights3-improvement-30-1.96.hdf5')
 
  history=model.fit_generator(
    gen(20, args.host, port=args.port,m=model),
    steps_per_epoch=50*1200/32.,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port,m=model),
    validation_steps=20*1200/32.,verbose=1,callbacks=callbacks_list)

  np.save('./loss_history/loss_danet_0',np.array(history.history['loss']))
  np.save('./loss_history/val_loss_danet_0',np.array(history.history['val_loss']))
