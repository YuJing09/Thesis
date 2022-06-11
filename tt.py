import os
import random
import numpy as np
import cv2
import time
import h5py
import random
path='/media/jinnliu/新增磁碟區/train/'
vpath=['/media/jinnliu/新增磁碟區/validation/']
traindir=os.listdir(path)
pathh=[path + k +'/' for k in traindir]
camerapath=[w+'camera.h5' for w in pathh]
vcamerapath=[w+'camera.h5' for w in vpath]
traindir=sorted(traindir,key=lambda x:int(x.replace('n','')))
label=dict()
imagepath=[]
batch=256
batchlabel=np.zeros([batch,1])
for i,j in enumerate(traindir):
    label[j]=i
for _ in range(batch):    
    randomtrain=random.choice(traindir)
    rimagepath=os.listdir(path+randomtrain+'/')
    rimage=random.choice(rimagepath)
    imagepath.append(path+randomtrain+'/'+rimage)
for b in range(batch):
    for l in label:
        if l in imagepath[b]:
           batchlabel[b]=label[l]
           #print(imagepath[b],l,label[l])
           pass
vlabel=[]
def mkcamera(p):
    
    for __ in p:
        
       #print(os.listdir(__))
        if   not os.path.exists(__+'camera.h5'):
         
           
          with h5py.File(__+'camera.h5','w') as ca:
               camera=os.listdir(__)
               with open('/media/jinnliu/新增磁碟區/ILSVRC2012_validation_ground_truth.txt','r') as f:
                 for line in f.readlines():
                   vlabel.append(line.strip('\n'))
               camera=[h for h in camera if ".JPEG" in h]
               
               camera=sorted(camera,key=lambda x : int(x[15:23]))
              
               
               cameracount=len(camera)
               ca.create_dataset('X',(cameracount,260,260,3))
             
               for ll in range(cameracount):
                 
                   img=cv2.imread(__+camera[ll])
                 
                   if img.shape[0]>img.shape[1]:
                      h=img.shape[1]
                      img=cv2.resize(img,(260,int(img.shape[0]*260/h)))
                      img=img[img.shape[0]//2-130:img.shape[0]//2+130,:]
                   else:
                      h=img.shape[0]
                      img=cv2.resize(img,(int(img.shape[1]*260/h),260))
                      img=img[:,img.shape[1]//2-130:img.shape[1]//2+130]
                   print('path:%s %d,label:%s '%(__+camera[ll],ll,vlabel[ll]))
                   ca['X'][ll]=img 	


#mkcamera(vpath)
#first = True
def concatenate(camera_names,label,val=False):
  
  lastidx = 0
  c5x=[]
  classes=[]
  for cword in camera_names:
    c5=h5py.File(cword,'r')
    
    x=c5['X']
    #print(type(x))
    #print(lastidx)
    c5x.append((lastidx,lastidx+x.shape[0],x))
    
    #print("x {} ".format(x.shape[0]))
    lastidx+=x.shape[0]
    if not val:
      for l in label:
        if l in cword:
          print(l)
          #print(l,label[l])
          classes+=[label[l]]*x.shape[0]
          break
    else:
      with open('/media/jinnliu/新增磁碟區/val.txt','r') as f:
        for line in f.readlines():
            classes.append(line[29:])
            #print(line[29:])
        classes=list(map(int,classes))
    #print("training on %d examples" % (x.shape[0]))
  #print(classes)
  #c5.close()
  #print(classes,lastidx)
  return (c5x,lastidx,classes)

def datagen(filter_files,label, time_len=1, batch_size=256,val=False):

    global first
    c5x,img_num,classes= concatenate(filter_files,label,val=val)
    classes=np.array(classes)
    
    classes_num=1000
    
    #print(classes)
    
    classes=np.eye(classes_num)[classes]
    X_batch = np.zeros((batch_size, time_len, 260, 260, 3), dtype='float32')
    #X_batch_uint=np.zeros((batch_size,time_len,260,260,3),dtype='uint8')
    classes_batch = np.zeros((batch_size, time_len,classes_num), dtype='float32')
    while True:
        
      t = time.time()
      count=0
      while count < batch_size:
        i=np.random.randint(0,img_num,1)
        for es,ee,x in c5x:
          if i>=es and i<ee:
            if not val:
              r=random.random()
              
              if r >= 0.5:
                
                  
                X_batch[count]=x[i[0]-es:i[0]-es+time_len][:,::-1,::-1]
                #X_batch_uint[count]=x[i[0]-es:i[0]-es+time_len]
              else:
                X_batch[count]=x[i[0]-es:i[0]-es+time_len][:,:,::-1]
                break
            else:
              
              X_batch[count]=x[i[0]-es:i[0]-es+time_len][:,:,::-1]
        classes_batch[count]=classes[i[0]:i[0]+time_len]
         
        count+=1
      t2=time.time()
      print(t2-t)
      
      yield (X_batch, classes_batch)



