import os
import random
import numpy as np
import cv2
import time
import h5py
import matplotlib.pyplot as plt
from modell import Ypgru

  

def concatenate(camera_names):
  #logs_names = [x.replace('camera320', 'pathdata') for x in camera_names]
  #lead_names = [x.replace('camera320', 'radardata') for x in camera_names]
  steering_names= [x.replace('camera320.h5','global_pose/frame_steers.npy') for x in camera_names]
  lastidx = 0
  c5x=[]
  steers=[]
  #model=Ypgru(2)
  #input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  #input2=tf.keras.Input(shape=(512),name='rnn')
  #adam=optimizers.Adam(lr=0.0001)
  #output=model.call(input1,input2)
  #model =tf.keras.Model(inputs=[input1,input2],outputs=output)
  #model.compile(optimizer=adam, loss=custom_loss)
  #model.load_weights('./saved_model/weights-improvement-01-123.11.hdf5')
  
  for cword, sword in zip(camera_names,steering_names):
    if os.path.isfile(cword):
      c5=h5py.File(cword, "r")
      t0=time.time()
        
        
      
        #velocity=t5['V'][:]
        #Path=t5['Path'][:]
        #path.append(Path)
      t1=time.time()
        #print('loadtime:',(t1-t0))
      steering=np.load(sword)
      t2=time.time()
        #leadOne=l5['LeadOne'][:]
      steers.append(steering[:-50])
      #print(len(steers[-1]),c5['X'].shape[0])
        #print(lastidx)
      x=c5['X']
      print(cword)
      c5x.append([lastidx,lastidx+c5['X'].shape[0],x])
        
        #print("x {} ".format(x.shape[0]))
      lastidx+=c5['X'].shape[0]
        
        
      
  
  steers=np.concatenate(steers,axis=0)
  print(np.sum(steers)/len(steers))
  steers=np.reshape(steers,[-1,1])
  
  print(lastidx)
  print(len(steers))
  print ("training on %d examples" % (lastidx))
  
  #print(lastidx)
  return c5x,lastidx,steers
def datagen(camera_files, max_time_len=10, batch_size=10):
  c5x,lastidx,steers=concatenate(camera_files)

  
  
  X_batch = np.zeros((batch_size, max_time_len, 160, 320, 3), dtype='uint8') 
  steering_batch = np.zeros((batch_size, max_time_len, 1), dtype='float32')
  #velocity_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
  while True:
    t=time.time()
    count=0
    #time_len=np.random.randint(1,max_time_len+1,1)[-1]
    time_len=1
    print(time_len)

    X_batch_t=X_batch[:,:time_len]
    steering_batch_t=steering_batch[:,:time_len]
    while count < batch_size:
      
      ri = np.random.randint(0, lastidx, 1)[-1]
    
      for fl,el,x in c5x:
        
          
        if fl <= ri and ri< el:
        
          #if el-ri<time_len:
          
          rj=np.random.randint(0,(el-fl)//5+1-time_len,1)[-1]
          #print(rj,outindex,ri[count])
            
          X_batch[count]=x[rj*5:(rj+time_len)*5:5] #if time_len>1
          
          
          steering_batch[count]=steers[fl+rj*5:fl+(rj+time_len)*5:5]
              
          #else:
            
          #X_batch_t[count]=x[ri-fl:ri-fl+time_len]
          #lead_path_batch_t[count]=out[ri:ri+time_len]
          break
      #print(lead_path_batch[count])
          #velocity_batch[count]=velocities[r][rj:rj+time_len]
      count+=1
   
    #print(lead_path_batch[2][:,-1])
    t2=time.time()
    #plt.imshow(a)
    #plt.show()
    #b=np.hstack(X_batch[1])
    #plt.imshow(b)
    #plt.show()
    #print(steering_batch_t)
    print("%5.2f ms" % ((t2-t)*1000.0),X_batch_t.shape,steering_batch_t.shape)
    yield (X_batch,steering_batch)
if __name__ == "__main__":
  from train_path_new import custom_loss
  from tensorflow.keras import optimizers
  import tensorflow as tf

  fi=os.listdir('/media/jinnliu/My Passport/')
#i=os.listdir('../1e563d9b-ab96-4b46-b374-c2bd73b3c7f0')
  fi=[i for i in fi if 'Chunk' in i]
  fi=['/media/jinnliu/My Passport/'+i +'/' for i in fi]

  print(fi)
  path_all=[]
  for f in fi:
    dir1=os.listdir(f)
    path=[f+ d +'/' for d in dir1]
    for ff in path:
      dir2=os.listdir(ff)
      #pathh=[ff + dd +'/video.hevc' for dd in dir2]
      pathh=[ff + dd +'/camera320.h5' for dd in dir2]
      #pathhl=list(map(lambda x: x.replace('camera','log'),pathh))
      for fff in pathh:
        #os.remove(fff)
        #os.remove(fff.replace('camera','log'))
        path_all.append(fff)
#print(path_all)
  #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
  model=Ypgru(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1,input2)
  model =tf.keras.Model(inputs=[input1,input2],outputs=output)

  concatenate(path_all[200:],model)

