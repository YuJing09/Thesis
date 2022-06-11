import numpy as np
import pygame
import matplotlib.pyplot as plt
import os
import cv2
from classifyvideo import path
pygame.init()
size=(320,160)
pygame.display.set_caption("label tool")
screen=pygame.display.set_mode(size,pygame.DOUBLEBUF)
camera_surface=pygame.surface.Surface(size,0,24).convert()
#path="/media/yp/新增磁碟區/classimg/"
#allimg=os.listdir(path)
#testpath=[path+i for i in allimg if 'png' in i][:]
#print(path)
count=0
la=[]
camerapath=[]
c=['0.jpg','100.jpg','200.jpg','300.jpg','400.jpg','500.jpg','600.jpg','700.jpg','800.jpg','900.jpg','1000.jpg','1100.jpg']
for i in path:
  cameradir=i.replace('camera320.h5','camera/')
  try:
    cc=os.listdir(cameradir)
  except:
    continue
  for j in c:
    if j in cc:
      camerapath.append(cameradir+j) 
print(len(camerapath))

if os.path.isfile('./label.txt'):
  f=open('./label.txt','r+')
  for i in f.readlines():
    
    la.append(int(i.split('\t')[1][0]))
    true=np.array(la)
    #print(np.sum(true))
    print(la)
    count=len(la)
else:
  f=open('./label.txt','w')


while count<len(camerapath):
  imgpath=camerapath[count]
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        exit(0)
    elif event.type == pygame.KEYDOWN :
      if event.key == pygame.K_0:
        label=0
        
        
        f.write(camerapath[count]+"\t%d\n"%(label))
        count+=1
        pygame.time.wait(100)
      if event.key == pygame.K_1:
        label=1
        
        f.write(camerapath[count]+"\t%d\n"%(label))  
        count+=1    
        pygame.time.wait(100)
        
  img=plt.imread(imgpath)
  
  img.astype(np.int32)
  img=img[:,:,:]
  print(count)
  #rect=np.array([[550,400],[640,400],[960,643],[256,643]])
  #rect=rect.reshape([4,1,2])
  #cv2.polylines(img,[rect],True,(0,255,255))
  pygame.surfarray.blit_array(camera_surface,img.swapaxes(0,1))
  screen.blit(camera_surface,(0,0))
  pygame.display.flip()
f.close()
