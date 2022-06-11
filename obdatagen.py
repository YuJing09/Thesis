import os
import random
import numpy as np
import cv2
import time
import h5py
import random
from matplotlib import pyplot as plt

trainimg_path='/media/jinnliu/1e563d9b-ab96-4b46-b374-c2bd73b3c7f0/darknet-master/scripts/coco/images/train2014/'
valimg_path='/media/jinnliu/1e563d9b-ab96-4b46-b374-c2bd73b3c7f0/darknet-master/scripts/coco/images/val2014/'
trainlabel_path='/media/jinnliu/1e563d9b-ab96-4b46-b374-c2bd73b3c7f0/darknet-master/scripts/coco/labels/train2014'
vallabel_path='/media/jinnliu/1e563d9b-ab96-4b46-b374-c2bd73b3c7f0/darknet-master/scripts/coco/labels/val2014/'

def concatenate(camera_names):
 
  lastidx = 0
  c5x=[]
  labels=[]
  
  c5=h5py.File(camera_names+'camera.h5','r')
  label_names=camera_names.replace('images', 'labels') 
  x=c5['X']
  label=h5py.File(label_names+'log.h5','r')  
  c5x.append((lastidx,lastidx+x.shape[0],x))
  l=label['label']
  
    
  lastidx+=x.shape[0]
  print(x.shape,l.shape)
  return (c5x,lastidx,l)
def datagen(filter_files, time_len=1, batch_size=5):


    c5x,img_num,labels= concatenate(filter_files)
    
    #print(img_num)
    
    
    
    
    
    X_batch = np.zeros((batch_size, time_len, 512, 512, 3), dtype='float32')
      #X_batch_uint=np.zeros((batch_size,time_len,260,260,3),dtype='uint8')
    label_batch = np.zeros((batch_size,time_len,100,5),dtype='float32')
    while True:
        
      t = time.time()
      count=0
      ll=[]
      while count < batch_size:
        i=np.random.randint(0,img_num,1)
        for es,ee,x in c5x:
          if i>=es and i<ee:
            
            r=random.random()
            l=[]
            
            #print(labels)  
            if r >= 0.5:
                
                 
              X_batch[count]=x[i[0]-es:i[0]-es+time_len][:,:,::-1]
             
              for w in labels[i[0]]:
            
                if -1 not in w:
                  #print(w)
                  a=512-w[0]
                  b=512-w[2]
                  w[0]=b
                  w[2]=a
                  l.append(w)
                  
                else:
                  l.append(w)
              l=np.array(l)
              #print(l)
              
              label_batch[count][-1]=np.array(l)
              
            else:
              X_batch[count]=x[i[0]-es:i[0]-es+time_len][:,:]
              
              label_batch[count][-1]=labels[i[0]]
        #print(label_batch)      
           
   
        count+=1
      t2=time.time()
      #print(t2-t)
      #print(X_batch,label_batch)
      
      print(X_batch.shape,label_batch.shape) 
      yield (X_batch, label_batch)
if __name__ == '__main__':
  print(trainimg_path)
  
  t=datagen(trainimg_path)
  x,y=next(t)
  print(x,y.shape)
  #plt.subplot(2,1,1)
  plt.imshow(x[0][0].astype(np.int))
  #print(x.shape)
  #plt.subplot(2,1, 2)
  #plt.imshow(x[0][0][:,::-1].astype(np.int))
  plt.show()
