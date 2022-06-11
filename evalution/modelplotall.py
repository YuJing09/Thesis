import tensorflow as tf
import argparse
import json 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from modell_transform import Ypgru,Ypgru2
from modell2 import Ypdanet,Ypreg
from server import client_generator
from server_path import path
from modell_steer import Ypregsteer
from tools.lib.logreader import LogReader
import scipy
from lib.camera import img_from_device, denormalize, view_frame_from_device_frame
from tools.lib.framereader import FrameReader
from car_calc_curvature import calc_lookahead_offset_civic
from common.transformations.camera import FULL_FRAME_SIZE, eon_intrinsics
from common.transformations.model import (MODEL_CX, MODEL_CY, MODEL_INPUT_SIZE,
                                          get_camera_frame_from_model_frame)
from tools1.replay.lib.ui_helpers import (draw_lead_car, draw_lead_on, draw_mpc,
                                         draw_path, draw_steer_path,
                                         init_plots, to_lid_pt, warp_points)
                                       
_BB_OFFSET = 0,0
_BB_SCALE = 1164/640.
_BB_TO_FULL_FRAME = np.asarray([
    [_BB_SCALE, 0., _BB_OFFSET[0]],
    [0., _BB_SCALE, _BB_OFFSET[1]],
    [0., 0.,   1.]])
_FULL_FRAME_TO_BB = np.linalg.inv(_BB_TO_FULL_FRAME)
CalP = np.asarray([[0, 0], [MODEL_INPUT_SIZE[0], 0], [MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]], [0, MODEL_INPUT_SIZE[1]]])
vanishing_point = np.asarray([[MODEL_CX, MODEL_CY]])  
class CalibrationTransformsForWarpMatrix(object):
  def __init__(self, model_to_full_frame, K, E):
    self._model_to_full_frame = model_to_full_frame
    self._K = K
    self._E = E

  
  def model_to_bb(self):
      
      return _FULL_FRAME_TO_BB.dot(self._model_to_full_frame)

  
  def model_to_full_frame(self):
    return self._model_to_full_frame

  
  def car_to_model(self):
    return np.linalg.inv(self._model_to_full_frame).dot(self._K).dot(
      self._E[:, [0, 1, 3]])

  
  def car_to_bb(self):
    return _FULL_FRAME_TO_BB.dot(self._K).dot(self._E[:, :])


# ***** get perspective transform for images *****
os.environ['CUDA_VISIBLE_DEVICES'] = " "
def draw_path1(device_path, img, width=0, height=1.2, fill_color=(174,48,96), line_color=(30,144,255),line=True,point=False):
  device_path_l = device_path + np.array([0, 0, height])                                                                    
  device_path_r = device_path + np.array([0, 0, height])                                                                    
  device_path_l[:,1] -= width                                                                                               
  device_path_r[:,1] += width 

  img_points_norm_l = img_from_device(device_path_l)
  img_points_norm_r = img_from_device(device_path_r)
  img_pts_l = denormalize(img_points_norm_l)
  img_pts_r = denormalize(img_points_norm_r)

  # filter out things rejected along the way
  valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
  img_pts_l = img_pts_l[valid].astype(int)
  img_pts_r = img_pts_r[valid].astype(int)
  filter_index=(np.logical_and.reduce((np.all(  # pylint: disable=no-member
    img_pts_l > 0, axis=1), img_pts_l[:, 0] < img.shape[1] - 1, img_pts_l[:, 1] <
                                                  img.shape[0] - 1))) 
  uv_pts=img_pts_l[filter_index]
  if point:
    for i, j  in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
    
    
        img[uv_pts[:, 1] + i, uv_pts[:, 0] + j] = (255,0,0)
  
  if line:
    for i in range(1, len(img_pts_l)):
      u1,v1,u2,v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
      u3,v3,u4,v4 = np.append(img_pts_l[i], img_pts_r[i])
      pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
      cv2.fillPoly(img,[pts],fill_color)
      cv2.polylines(img,[pts],True,line_color)
def draw_radar(device_radar, img, width=1, height=0.2, line_color=(155,144,0)):
  device_radar_l = device_radar + np.array([0, 0, height])                                                                    
  device_radar_r = device_radar + np.array([0, 0, height]) 
  device_radar_l[:,1] -= width                                                                                               
  device_radar_r[:,1] += width 
  img_points_norm_l = img_from_device(device_radar_l)
  img_points_norm_r = img_from_device(device_radar_r)
  img_pts_l = denormalize(img_points_norm_l)
  img_pts_r = denormalize(img_points_norm_r)
  valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
  img_pts_l = img_pts_l[valid].astype(int)
  img_pts_r = img_pts_r[valid].astype(int)
  
  u1,u2,u3,u4=np.append(img_pts_l[0],img_pts_r[0])
  v1=u1
  v3=u3
  v2=u2
  print(u1,u2,u3,u4)
  pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
  cv2.fillPoly(img,[pts],fill_color)
  #print(u1,u2,u3,u4)
def getvalidpoint(x):
  #print(x)
  front=np.arange(50)+1
  points=x[0,:50]
  valid_len=np.clip(int(x[0,50]*10),0,50)
  w=np.polyfit(front[:valid_len],points[:valid_len],3)
  fitpoint=w[3]+w[2]*front+w[1]*front**2+w[0]*front**3
  return (front[:valid_len],fitpoint[:valid_len])
if __name__=="__main__":
  model=Ypdanet(2)
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  
  input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1)
  model =tf.keras.Model(inputs=[input1],outputs=output)
  model1=Ypgru(2)
  output1=model1.call(input1,input2)
 
  model1 =tf.keras.Model(inputs=[input1,input2],outputs=output1)
  model2=Ypreg(2)
  output2=model2.call(input1)
  model2=tf.keras.Model(inputs=[input1],outputs=output2)
  #input2=tf.keras.Input(shape=(512),name='rnn')
  x=np.arange(50)+1
  model3=Ypregsteer(2)
  output3=model3.call(input1)
  model3=tf.keras.Model(inputs=[input1],outputs=output3) 
  #model_d=Discriminator((160,320,3))
  #output4=model_d.call(input1)
  #model_d=tf.keras.Model(inputs=[input1],outputs=output4)
  
  #model3=Ypgru2(2)
  #output3=model3.call(input1,input2)
  #model3=tf.keras.Model(inputs=[input1,input2],outputs=output3)
  model.load_weights('./saved_model/weights3-improvement-02-2.01.hdf5')
  model1.load_weights('./saved_model/weights-gru-improvement-85-2.08.hdf5')
  model2.load_weights('./saved_model/weights-reg-improvement-30-2.02.hdf5')
  model3.load_weights('./saved_model/steer-weights-improvement-159-0.59.hdf5')
  #model_d.load_weights('./saved_model/discriminator.keras')
  #model3.load_weights('./saved_model/weights-v-improvement-01-2.96.hdf5')
  data_len=len(path)
  test_path=path[-1*int(0.25*data_len):]
  np.random.seed(99)
  example_path=np.random.choice(test_path,30)
  allimg=[]
  front=np.arange(50)+1
  img1=[]
  img2=[]
  num=0
  print(example_path)
  intrinsic_matrix = eon_intrinsics
  #example_path=['../Chunk_1/b0c9d2329ad1606b_2018-07-27--06-50-48/12/camera320.h5']
  for i in example_path[:100]:
    rnn_state=np.zeros([1,512])
    with h5py.File(i,'r') as c:
      try:
        log_path=i.replace('camera320.h5','log.h5')
        speed_path=i.replace('camera320.h5','global_pose/frame_velocities')
        steer_path=i.replace('camera320.h5','global_pose/frame_steers.npy')
        path_data=i.replace('camera320.h5','pathdata.h5')
        radarpath=i.replace('camera320','radardata')
        hevcpath=i.replace('camera320.h5','video.hevc')
        raw_log=i.replace('camera320.h5','raw_log.bz2')
        
        fr=FrameReader(hevcpath)
        
        speed=np.load(speed_path)
        steer=np.load(steer_path)
        log=h5py.File(log_path,'r')
        extrinsicmartix=log['extrinsicmatrix']
        path=h5py.File(path_data,'r')
        l5=h5py.File(radarpath,'r')
        
        P=path['Path']
        img=c['X']
        leadone=l5['LeadOne']
        
      except:
        continue
      
      random_img=np.random.randint(0,len(img),1)[0]
      #random_img=8
      random_img_raw=fr.get(random_img, pix_fmt='rgb24')[0]
      img_d=cv2.resize(random_img_raw,(320,160))
      print(img_d.shape,img[0].shape)
      img_r=img[random_img:random_img+1][:,:,:]
      #prob_lead=model_d.predict(x=[img_r])[0]
      #print(prob_lead)
      #print(extrinsicmartix[:].shape)
      extrinsic_matrix = extrinsicmartix[random_img]
      #print(extrinsic_matrix)
      ke = intrinsic_matrix.dot(extrinsic_matrix)
      warp_matrix = get_camera_frame_from_model_frame(ke)
      calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)
      #print(calibration.car_to_bb())
      #cpw = warp_points(CalP, calibration.model_to_bb())
      #vanishing_pointw = warp_points(vanishing_point, calibration.model_to_bb())
      #print(vanishing_pointw)
      #print(type(img_r))
      #img_r=cv2.warpAffine(img_r, (_BB_TO_FULL_FRAME)[:2],(640, 480), dst=img_r, flags=cv2.WARP_INVERSE_MAP)
      
      imgg=cv2.warpAffine(random_img_raw, (_BB_TO_FULL_FRAME)[:2],(640, 480), dst=random_img_raw, flags=cv2.WARP_INVERSE_MAP)
      
      predictions_a=model.predict(x=[img_r])
      
      stds=np.exp(predictions_a[0,51:101])
      #print(predictions_a[0,51:101])
      radarpredict=predictions_a[0,101:]
      radar_x=radarpredict[0]*10
      radar_y=radarpredict[1]
      #if leadone[random_img][0] !=0 and leadone[random_img][1]!=0:
      x=np.array([radar_x,radar_x,radar_x,radar_x])
      y=np.array([radar_y-1,radar_y+1,radar_y+1,radar_y-1])
      z=np.array([0.5,0.5,-0.5,-0.5])
      #print(radar_x,radar_y)
        #device_path=np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=-1)
        #print(device_path)
      draw_path(y,x,z,(255,255,0),imgg,calibration,None,None)
        
      print(i,num)
      radar_xyz=np.array([[leadone[random_img][0],leadone[random_img][1]*-1,0],[leadone[random_img][0]+1,leadone[random_img][1]*-1,0]])
      #draw_path(radar_xyz,random_img_raw,width=1,line_color=(255,255,0))
      x1,y1=getvalidpoint(predictions_a)
      #print(radar_x,radar_y,leadone[random_img],y1,num)
      z1=np.zeros([len(x1)])
      device_path1=np.concatenate([x1.reshape(-1,1),y1.reshape(-1,1),z1.reshape(-1,1)],axis=-1)
      #print(y1,x1,z1)
      print(device_path1)
      draw_path1(device_path1,imgg,line=False,point=True)
      #draw_path(y1*-1+0.4,x1,z1,(255,0,0),imgg,calibration,None,line=False)
      #draw_path((y1*-1+0.4)[:],x1[:],z1[:],(255,0,0),imgg,calibration,None,line=False,stds=stds[:])
      #draw_path(y1*-1+0.4-stds,x1,z1,(0,0,200),imgg,calibration,None,line=False)
      #print(device_path1,radar_xyz)
      device_path2=device_path1.copy()
      device_path2[:,1]+=stds
      device_path3=device_path1.copy()
      device_path3[:,1]-=2*stds
      draw_path1(device_path2,imgg)
      draw_path1(device_path3,imgg)
      #draw_path(device_path2[:,1]*-1,device_path2[:,0],(255,0,0),imgg,calibration,None)
      #draw_path(device_path1[:,1]*-1,device_path1[:,0],(30,144,255),imgg,calibration,None,None)
      #draw_path(device_path3[:,1]*-1,device_path3[:,0],(30,144,255),imgg,calibration,None,None)
      #draw_path1(device_path1,imgg)
      cv2.imwrite('./results4/aa%d.png'%(num),imgg[:,:,::-1])
      #device_path3=device_path1[:,1]-=stds
      #device_path1[:,1]-=2*stds
      #draw_path(device_path1,random_img_raw)
      
      #device_path1[:,1]+=stds
      #draw_path(device_path1,random_img_raw,line_color=(100,0,0))
      #device_path1[:,1]-=2*stds
      #draw_path(device_path1,random_img_raw,line_color=(100,0,0))
      #cv2.imwrite('./results4/result%d.png'%(num),random_img_raw[:,:,::-1])
      #print(device_path1,stds,radarpredict)
      
      
      num+=1
  
