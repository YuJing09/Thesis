import tensorflow as tf
import argparse
import json 
import os
import cv2
import numpy as np
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from modell2 import Ypgru,Ypdanet,Ypreg
from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from tensorflow.keras import optimizers

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
    p=m.predict(x=X_tt)
    loss=custom_loss(Y_t,p) 
    
    yield X_tt, Y_t
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
    s=s+loss_v/1000.+bce+valid_len_loss+loss_p+lead_all_loss
    
  all_loss=s/tf.cast(batch_size,dtype=tf.float32)
  
  return all_loss
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5558, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=30, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')

  args = parser.parse_args()
 
  model=Ypreg(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1)
  model =tf.keras.Model(inputs=[input1],outputs=output)
  
  filepath="./saved_model/weights-reg-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
  mode='min')
  callbacks_list = [checkpoint]
  adam=optimizers.Adam(lr=0.0001)
  model.load_weights('./saved_model/weights-reg-improvement-20-2.79.hdf5',by_name=True)
  #for layer in model.layers:
   # layer.trainable = False
    #print(layer.name)
   # if 'path_weight' in layer.name:
    #  layer.trainable=True
    #print(layer.name,layer.trainable)
  #model.save('ypdanet.h5')
  model.summary()
  #print(x.shape,y.shape)
  #model.summary()
  #gg=gen(20, args.host, port=args.port,m=model)
  model.compile(optimizer=adam, loss=custom_loss)
  history=model.fit_generator(
    gen(20, args.host, port=args.port,m=model),
    steps_per_epoch=20*1200/25.,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port,m=model),
    validation_steps=20*1200/25.,verbose=1,callbacks=callbacks_list)
