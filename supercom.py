from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json 
import os
import matplotlib.pyplot as plt
from server_path import path
import h5py
import cv2 
import lib.orientation as orient
import lib.coordinates as coord
from modell_transform import Ypgru,Ypgru2
from modell2 import Ypdanet,Ypreg
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
from tools.lib.framereader import FrameReader
import cv2
import sys
MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.
data_len=len(path)
#test_path=path[-1*int(0.25*data_len):]
camerafile = path[-1*int(0.25*data_len):]
LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

supercombo = load_model('models/supercombo.keras')

model_danet=Ypdanet(2)
input1=tf.keras.Input(shape=(160,320,3),name='img')
  
input2=tf.keras.Input(shape=(512),name='rnn')
output=model_danet.call(input1)
model_danet =tf.keras.Model(inputs=[input1],outputs=output)
model1=Ypgru(2)
output1=model1.call(input1,input2)
 
model_gru =tf.keras.Model(inputs=[input1,input2],outputs=output1)
model2=Ypreg(2)
output2=model2.call(input1)
model_reg=tf.keras.Model(inputs=[input1],outputs=output2)

model_danet.load_weights('./weights3-improvement-02-2.01.hdf5')
model_gru.load_weights('./weights-gru-improvement-85-2.08.hdf5')
model_reg.load_weights('./weights-reg-improvement-30-2.02.hdf5')
#model3.load_weights('./saved_model/steer-weights-improvement-159-0.59.hdf5')
def RMSE(x,y):
  y=np.array(y)
  
  
  return (((x-y)**2))
def getvalidpoint(x):
  #print(x)
  front=np.arange(50)+1
  points=x[0,:50]
  valid_len=np.clip(int(x[0,50]*10),0,50)
  w=np.polyfit(front[:valid_len],points[:valid_len],3)
  fitpoint=w[3]+w[2]*front+w[1]*front**2+w[0]*front**3
  return (front,fitpoint)
def fit(x,y):
 
  f=np.arange(50)+1
  
  a0,a1,a2,a3=np.polyfit(x,y,3)
  yy=a3+f*a2+f**2*a1+a0*f**3
  
  return yy

def ADE(x,y,l):
  return np.sum((x[:l]-y[:l])**2)/l
def FDE(x,y,l):
  return np.abs((x[l-1]-y[l-1])**2)
def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1


ade_all=np.zeros(4)
fde_all=np.zeros(4)
lead_rmse_all=np.zeros([4,4])
s=0
ss=0
for i in camerafile:
  state = np.zeros((1,512))
  desire = np.zeros((1,8))
  rnn_state=np.zeros((1,512))
  imgs = []
  with h5py.File(i,'r') as c:
    speed_path=i.replace('camera320.h5','global_pose/frame_velocities')
    steer_path=i.replace('camera320.h5','global_pose/frame_steers.npy')
    pose_ori=i.replace('camera320.h5','global_pose/frame_orientations')
    path_data=i.replace('camera320.h5','pathdata.h5')
    radarpath=i.replace('camera320.h5','radardata.h5')
    log_path=i.replace('camera320.h5','log.h5')
    try:
      log=h5py.File(log_path,'r')
      extrinsicmartix=log['extrinsicmatrix']
     
      path=h5py.File(path_data,'r')
      l5=h5py.File(radarpath,'r')
    except:
      continue
    P=path['Path'] 
    img320=c['X']
    
    
    cap = cv2.VideoCapture(i.replace('camera320.h5','video.hevc'))
    fr=FrameReader(i.replace('camera320.h5','video.hevc'))
    for ii in tqdm(range(fr.frame_count-50)):
      ret, frame = cap.read()
  
      img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

      imgs.append(img_yuv.reshape((874*3//2, 1164)))
    imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    for iii, img in tqdm(enumerate(imgs)):
      imgs_med_model[iii] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
    frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
    frame_orientations = np.load(pose_ori)    
    leadOne=l5['LeadOne'][:]
    out=np.concatenate((P,leadOne),axis=-1)
    out[:,50:51]/=10.
    lead_ground_truth=out[:,51:55]
    lead_prob=out[:,-1]
    cap = cv2.VideoCapture(i.replace('camera320.h5','video.hevc'))
  #print(frame_tensors)
    for j in range(0,len(frame_tensors)-1,5):
     # print(j,len(frame_tensors-1))
      #supercombo
      
      ground_truth_p=out[j][:50]
      l=np.clip(int(out[j][50]*10),0,50)
      if np.sum(extrinsicmartix[j]) !=0:
        extrinsic_matrix = extrinsicmartix[j]
      front=np.arange(192)
      lead_truth=lead_ground_truth[j]
      z=np.zeros([192])
  #print(i)
      inputs = [np.vstack(frame_tensors[j:j+2])[None], desire, state]
      outs = supercombo.predict(inputs)
      parsed = parser(outs)
  # Important to refeed the state
      state = outs[-1]
      path_op=parsed['path'][0]*-1
      lead_op=parsed['lead_xyva'][0]
      
      """
      device_path=np.column_stack([front,path_op,z])
      lead_op=parsed['lead_xyva'][0]
      #print(lead_prob[j])
    
  #front=np.arange(len(path_op))
      car_path=extrinsic_matrix[:, [0, 1, 3]].dot(device_path.T)
      x=car_path[2]
      y=car_path[0]
      
      if l==0:
        continue
      
      index=np.argmin(np.abs(x-l))
      
      #print(x)
      fitsupercombo=fit(x[:index+1],y[:index+1])
  #print(fitsupercombo,ground_truth_p,l)
  """
      ade_supercombo=ADE(path_op,ground_truth_p,l)
      fde_supercombo=FDE(path_op,ground_truth_p,l)
   
    
    
    #opgru
      rnn_state=model_gru.predict(x=[img320[j:j+1],rnn_state])[0:1,106:]
    
      predictions_gru=model_gru.predict(x=[img320[j:j+1],rnn_state])
      y_gru=predictions_gru[0,:50]
      #x_gru,y_gru=getvalidpoint(predictions_gru)
      lead_gru=predictions_gru[-1,101:105]
      lead_gru[0]*=10
      #print(y_gru,'\n',car_path[2],car_path[1],car_path[0])
      ade_gru=ADE(y_gru,ground_truth_p,l)
      fde_gru=FDE(y_gru,ground_truth_p,l)
      
    #OPAN
    
      predictions_danet=model_danet.predict(x=[img320[j:j+1]])
      #x_danet,y_danet=getvalidpoint(predictions_danet)
      y_danet=predictions_danet[0,:50]
      lead_danet=predictions_danet[-1,101:105]
      lead_danet[0]*=10
      
      ade_danet=ADE(y_danet,ground_truth_p,l)
      fde_danet=FDE(y_danet,ground_truth_p,l)
      
    #OPPN  
      predictions_reg=model_reg.predict(x=[img320[j:j+1]])
      #x_reg,y_reg=getvalidpoint(predictions_reg)
      y_reg=predictions_reg[0,:50]
      lead_reg=predictions_reg[-1,101:105]
      lead_reg[0]*=10
      
      ade_reg=ADE(y_reg,ground_truth_p,l)
      fde_reg=FDE(y_reg,ground_truth_p,l)
    #ALL predict 
      if ade_supercombo<100: 
        ade_all+=ade_supercombo,ade_gru,ade_reg,ade_danet
        fde_all+=fde_supercombo,fde_gru,fde_reg,fde_danet
        #print(j)
        s+=1
      if lead_prob[j]:
        lead_rmse_all[0]+=RMSE(lead_op,lead_truth)
        lead_rmse_all[1]+=RMSE(lead_gru,lead_truth)
        lead_rmse_all[2]+=RMSE(lead_danet,lead_truth)
        lead_rmse_all[3]+=RMSE(lead_reg,lead_truth)
        ss+=1
    
    print(i+'\n')  
    print('ADE:',ade_all/s,'\n','FDE:',fde_all/s,'\n','LEAD:',lead_rmse_all/ss,'\n')
      
