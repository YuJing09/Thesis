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
from modell_steer import Ypregsteer
from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
)

def gen(hwm, host, port,m):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y= tup
    batch_size=X.shape[0]
    Y_t = Y[:, -1]
    X_tt= X[:,-1]
    rnn_state=np.zeros([batch_size,512])
    for i in range(X.shape[1]-1):
    
      
      #Y_t = Y[:, i]
        
        
      X_t = X[:, i]
   
      
      #rnn_statee=m.predict(x=[X_t,rnn_state])
      #rnn_state=rnn_statee[:,106:]
      
      
        
      
    #p=m.predict(x=[X_tt,rnn_state])
    #loss=custom_loss(Y_t,p) 
    
    yield [X_tt], Y_t
def custom_loss(y_true, y_pred,delta=2.):
  batch_size=tf.shape(y_true)[0]
  
  
  true_steer=y_true[:,-1]
  pred_steer=y_pred[:,0]
  
  
  #true_steer_std=tf.square(true_steer-pred_steer)
 
  #pred_steer_std=tf.math.exp(y_pred[:,1])
  
  #huber loss
  
  error_steer=tf.abs(tf.subtract(true_steer,pred_steer))
  #error_steer_std=tf.abs(tf.subtract(true_steer_std,pred_steer_std))
  

  condition_steer=tf.less(error_steer,delta)
  #condition_steer_std=tf.less(error_steer_std,delta)
  
    
  small_res_steer=0.5 * tf.square(error_steer)  #mse
  #small_res_steer_std=0.5 * tf.square(error_steer_std) #mse
  
    
  large_res_steer= delta * error_steer - 0.5 * tf.square(delta) 
  #large_res_steer_std= delta * error_steer_std - 0.5 * tf.square(delta) 
  
 
  steer_loss=tf.where(condition_steer,small_res_steer,large_res_steer)
  #steer_std_loss=tf.where(condition_steer_std,small_res_steer_std,large_res_steer_std)
  
  
  loss=steer_loss 
    #all_loss=points_all_loss
    
    #
    #print(points_all_loss,points_std_all_loss,lead_all_loss,lead_std_all_loss,valid_len_loss)
    
  

  
  return loss
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5558, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=160, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')

  args = parser.parse_args()
 
  model=Ypregsteer(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  #input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1)
  model =tf.keras.Model(inputs=[input1],outputs=output)
  
  adam=optimizers.Adam(lr=0.0001)
  filepath="./saved_model/steer-weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
  callbacks_list = [checkpoint]
 
 
  #print(x.shape,y.shape)
  #model.summary()
  
  model.compile(optimizer=adam, loss=custom_loss)
  #model.load_weights('./saved_model/steer-weights-improvement-02-3.06.hdf5')
 
  history=model.fit_generator(
    gen(20, args.host, port=args.port,m=model),
    steps_per_epoch=50*1200/32.,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port,m=model),
    validation_steps=25*1200/32.,verbose=1,callbacks=callbacks_list)
  print("Saving model weights and configuration file.")

  
  
  np.save('./loss_history/loss_steer',np.array(history.history['loss']))
  np.save('./loss_history/val_loss_steer',np.array(history.history['val_loss']))
