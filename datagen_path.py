import os
import random
import numpy as np
import cv2
import time
import h5py
import matplotlib.pyplot as plt
from modell import Ypgru

  

def concatenate(camera_names):
  logs_names = [x.replace('camera320', 'pathdata') for x in camera_names]
  lead_names = [x.replace('camera320', 'radardata') for x in camera_names]
  lastidx = 0
  c5x=[]
  path=[]
  #model=Ypgru(2)
  #input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  #input2=tf.keras.Input(shape=(512),name='rnn')
  #adam=optimizers.Adam(lr=0.0001)
  #output=model.call(input1,input2)
  #model =tf.keras.Model(inputs=[input1,input2],outputs=output)
  #model.compile(optimizer=adam, loss=custom_loss)
  #model.load_weights('./saved_model/weights-improvement-01-123.11.hdf5')
  lead=[]
  for cword, tword , lword in zip(camera_names, logs_names,lead_names):
    if os.path.isfile(lword):
      with h5py.File(tword, "r") as t5:
        t0=time.time()
        
        c5=h5py.File(cword,'r')
        l5=h5py.File(lword,'r')
      
        #velocity=t5['V'][:]
        Path=t5['Path'][:]
        path.append(Path)
        t1=time.time()
        #print('loadtime:',(t1-t0))
      
        t2=time.time()
        leadOne=l5['LeadOne'][:]
        lead.append(leadOne)
        #print(lastidx)
        x=c5['X']
        print(cword)
        c5x.append([lastidx,lastidx+c5['X'].shape[0],x])
        
        #print("x {} ".format(x.shape[0]))
        lastidx+=c5['X'].shape[0]
        
        
      
  #print(c5x)
  #print(path)
  path=np.concatenate(path,axis=0)
  a=[]
  for i in path:
    a.append(np.sum(i[:int(i[-1]*10)]))
  #thesis Table 4.1   
  print(np.sum(a)/len(a))
    
  
  #velocities=np.array(velocities)
  lead=np.concatenate(lead,axis=0)
  
  print(len(lead))
  
  
  #path+lead
  out=np.concatenate((path,lead),axis=-1)
  out[:,50:52]/=10
  
  print(np.max(out[:,51]))
  print(np.sum(out[:,55]))
  #c=0
  #rnn_state=np.zeros([1,512])
  
 # for i,j,x in c5x:
  
  #  while i<=23796 and 23796<j:
      #print(out[23796])
   #   x_t=x[23796-i-100:23796-i]
    #  rnn_state=np.zeros([1,512])
     # Y_t=out[23796-100:23796]
      #for i in range(100):
      #  x=x_t[i:i+1]
      #  y=Y_t[i:i+1]
        
       # p=model.predict([x,rnn_state])
       # rnn_state=p[:,110:]
      #loss=custom_loss(y,p)
      #print(loss)
      #break
  print ("training on %d examples" % (lastidx))
  
  #print(lastidx)
  return c5x,lastidx,out
def datagen(camera_files, max_time_len=15, batch_size=10):
  c5x,lastidx,out=concatenate(camera_files)

  
  
  X_batch = np.zeros((batch_size, max_time_len, 160, 320, 3), dtype='uint8') 
  lead_path_batch = np.zeros((batch_size, max_time_len, 51+5), dtype='float32')
  #velocity_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
  while True:
    t=time.time()
    count=0
    #time_len=np.random.randint(1,max_time_len+1,1)[-1]
    time_len=max_time_len
    #print(time_len)
    X_batch_t=X_batch[:,:time_len]
    lead_path_batch_t=lead_path_batch[:,:time_len]
    while count < batch_size:
      
      #print(ri)
      
      ri = np.random.randint(0, lastidx, 1)[-1]
    
      for fl,el,x in c5x:
        
          
        if fl <= ri and ri< el:
        
          #if el-ri<time_len:
          
          rj=np.random.randint(0,(el-fl)//5+1-time_len,1)[-1]
          #print(rj,outindex,ri[count])
            
          X_batch_t[count]=x[rj*5:(rj+time_len)*5:5] #if time_len>1
          
          
          lead_path_batch_t[count]=out[fl+rj*5:fl+(rj+time_len)*5:5]
              
          #else:
            
          #X_batch_t[count]=x[ri-fl:ri-fl+time_len]
          #lead_path_batch_t[count]=out[ri:ri+time_len]
          break
      #print(lead_path_batch[count])
          #velocity_batch[count]=velocities[r][rj:rj+time_len]
      count+=1
   
    #print(lead_path_batch[2][:,-1])
    t2=time.time()
    #a=np.hstack(X_batch[2])
    #plt.imshow(a)
    #plt.show()
    #b=np.hstack(X_batch[1])
    #plt.imshow(b)
    #plt.show()
    
    print("%5.2f ms" % ((t2-t)*1000.0),X_batch_t.shape,lead_path_batch_t.shape)
    yield (X_batch_t,lead_path_batch_t)
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

