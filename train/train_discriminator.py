import tensorflow as tf
import numpy as np
import h5py
import random
import json
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
class Discriminator(tf.keras.Model):
  def __init__(self,shape):
    super().__init__()
    self.shape=shape
    self._build()
  def _build(self):
    self.Norma=tf.keras.layers.Lambda(lambda x: x/127.5-1,input_shape=self.shape,output_shape=self.shape)
    self.convd0=tf.keras.layers.Conv2D(filters=16,kernel_size=[3,3],strides=[2,2],kernel_initializer='normal',data_format='channels_last',use_bias=True,name='conv2d_0')
    self.convd1=tf.keras.layers.Conv2D(filters=32,kernel_size=[3,3],strides=[2,2],kernel_initializer='normal',data_format='channels_last',use_bias=True,name='conv2d_1')
    self.convd2=tf.keras.layers.Conv2D(filters=64,kernel_size=[3,3],strides=[2,2],kernel_initializer='normal',data_format='channels_last',use_bias=True,name='conv2d_2')
    self.convd3=tf.keras.layers.Conv2D(filters=128,kernel_size=[3,3],strides=[2,2],kernel_initializer='normal',data_format='channels_last',use_bias=True,name='conv2d_3')
    #self.convd4=tf.keras.layers.Conv2D(filters=256,kernel_size=[3,3],strides=[2,2],kernel_initializer='normal',data_format='channels_last',use_bias=True,name='conv2d_4')
    self.flatten=tf.keras.layers.Flatten()
    self.dropout0=tf.keras.layers.Dropout(0.2)
    self.dropout1=tf.keras.layers.Dropout(0.5)
    self._relu_fn = tf.keras.layers.ELU()
    self.bn0=tf.keras.layers.BatchNormalization(axis=-1,epsilon=1e-5,name='bn_0')
    self.bn1=tf.keras.layers.BatchNormalization(axis=-1,epsilon=1e-5,name='bn_1')
    self.bn2=tf.keras.layers.BatchNormalization(axis=-1,epsilon=1e-5,name='bn_2')
    self.bn3=tf.keras.layers.BatchNormalization(axis=-1,epsilon=1e-5,name='bn_3')
    self.bn4=tf.keras.layers.BatchNormalization(axis=-1,epsilon=1e-5,name='bn_4')
    self.dense1=tf.keras.layers.Dense(512)
    self.dense=tf.keras.layers.Dense(1)
    self.sigmoid=tf.keras.activations.sigmoid
  def call(self,x):
    x=self.Norma(x)
    x=self.convd0(x)
    x=self.bn0(x)
    x=self._relu_fn(x)
    x=self.convd1(x)
    x=self.bn1(x)
    x=self._relu_fn(x)
    x=self.convd2(x)
    x=self.bn2(x)
    x=self._relu_fn(x)
    x=self.convd3(x)
    x=self.bn3(x)
    x=self._relu_fn(x)
    
    x=self.flatten(x)
    
    x=self.dropout0(x)
    x=self._relu_fn(x)
    out=self.dense(x)
    out=self.sigmoid(out)

    return out
if __name__=="__main__":
  imgpath=[]
  label=[]
  with open('label.txt','r') as f:
    for lines in f.readlines():
      a,b=lines.split()
      imgpath.append(a)
      label.append(b)
  
 # t=list(zip(img,car))
  imgall=[]
 # random.shuffle(t)
  for i in imgpath:
    img=cv2.imread(i)
    imgall.append(img)
  
 
  label=list(map(int,label))
  
 # print(img)
  #img_v=f_v['X'][:]
  #car_v=f_v['car'][:]
  #f.close()
  #f_v.close()
  #all_img=np.concatenate([img,img_v],axis=0)
  #all_car=np.concatenate([car,car_v],axis=0)
  
  imgall=np.array(imgall)
  print(imgall.shape)
  label=np.array(label).reshape([-1,1])
  print(imgall.shape,label.shape)
  adam=tf.keras.optimizers.Adam(lr=0.0001)
  #print(img_v.shape,car)
  saved_path='./saved_model/'
  Input=tf.keras.Input(shape=(160,320,3))
  shape=160,320,3
  model=Discriminator(shape)
  output=model.call(Input)
  model = tf.keras.Model(inputs=[Input],outputs=output)
  model.compile(optimizer=adam, loss='BinaryCrossentropy',metrics='binary_accuracy')
  model.summary()
  #model.load_weights(saved_path+'discriminator.keras')
  history=model.fit(imgall[:1800],label[:1800],verbose=1,epochs=60,shuffle=True,validation_data=(imgall[1800:],label[1800:]),validation_steps=500//32)
  np.save('./loss_history/loss.npy',np.array(history.history['loss']))
  np.save('./loss_history/val_loss.npy',np.array(history.history['val_loss']))
  np.save('./loss_history/binary_accuracy.npy',np.array(history.history['binary_accuracy']))
  np.save('./loss_history/val_binary_accuracy.npy',np.array(history.history['val_binary_accuracy']))
  
  print("Saving model weights and configuration file.")
  model.save_weights(saved_path+'discriminator.keras', True)
  with open(saved_path+'discriminator.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
