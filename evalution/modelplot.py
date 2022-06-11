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
intrinsic_matrix = eon_intrinsics
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
    return _FULL_FRAME_TO_BB.dot(self._K).dot(self._E[:, [0, 1, 3]])


# ***** get perspective transform for images *****
os.environ['CUDA_VISIBLE_DEVICES'] = " "
def draw_path1(device_path, img, width=0, height=1.2, fill_color=(128,0,255), line_color=(0,255,0)):
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

  for i in range(1, len(img_pts_l)):
    u1,v1,u2,v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
    u3,v3,u4,v4 = np.append(img_pts_l[i], img_pts_r[i])
    pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
    cv2.fillPoly(img,[pts],fill_color)
    cv2.polylines(img,[pts],True,line_color)
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
  #model3=Ypgru2(2)
  #output3=model3.call(input1,input2)
  #model3=tf.keras.Model(inputs=[input1,input2],outputs=output3)
  model.load_weights('./saved_model/weights3-improvement-02-2.01.hdf5')
  model1.load_weights('./saved_model/weights-gru-improvement-85-2.08.hdf5')
  model2.load_weights('./saved_model/weights-reg-improvement-30-2.02.hdf5')
  model3.load_weights('./saved_model/steer-weights-improvement-159-0.59.hdf5')
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
  for i in example_path:
    rnn_state=np.zeros([1,512])
    with h5py.File(i,'r') as c:
      try:
        speed_path=i.replace('camera320.h5','global_pose/frame_velocities')
        steer_path=i.replace('camera320.h5','global_pose/frame_steers.npy')
        path_data=i.replace('camera320.h5','pathdata.h5')
        radarpath=i.replace('camera320','radardata')
        hevcpath=i.replace('camera320.h5','video.hevc')
        log_path=i.replace('camera320','log')
        log=h5py.File(log_path,'r')
        extrinsicmartix=log['extrinsicmatrix']
        fr=FrameReader(hevcpath)
        speed=np.load(speed_path)
        steer=np.load(steer_path)
        path=h5py.File(path_data,'r')
        l5=h5py.File(radarpath,'r')
        P=path['Path']
        img=c['X']
        leadone=l5['LeadOne']
      except:
        continue
      
      random_img=np.random.randint(0,len(img),1)[0]
      #print(random_img)
      speed_r=speed[random_img]
      steer_r=steer[random_img]
      speed_r=np.linalg.norm(speed_r)
      y_steer=calc_lookahead_offset_civic(speed_r,steer_r,front)
      #print(steer_r)
      
      #print(speed_r,steer_r)
      random_img_raw=fr.get(random_img, pix_fmt='rgb24')[0]
      img_r=img[random_img:random_img+1][:,:,:,::-1]
      predict_steer=model3.predict(x=[img_r])
      y_steer_p=calc_lookahead_offset_civic(speed_r,predict_steer*-1,front,angle_offset=-1)
      extrinsic_matrix = extrinsicmartix[random_img]
      #print(extrinsic_matrix)
      ke = intrinsic_matrix.dot(extrinsic_matrix)
      warp_matrix = get_camera_frame_from_model_frame(ke)
      calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)
      #print(calibration.car_to_bb())
      cpw = warp_points(CalP, calibration.model_to_bb())
      vanishing_pointw = warp_points(vanishing_point, calibration.model_to_bb())
      #print(vanishing_pointw)
      #print(type(img_r))
      #img_r=cv2.warpAffine(img_r, (_BB_TO_FULL_FRAME)[:2],(640, 480), dst=img_r, flags=cv2.WARP_INVERSE_MAP)
      
      imgg=cv2.warpAffine(random_img_raw, (_BB_TO_FULL_FRAME)[:2],(640, 480), dst=random_img_raw, flags=cv2.WARP_INVERSE_MAP)
      #print(predict_steer,steer_r)
      path_r=P[random_img]
      radar_r=leadone[random_img]
      radar_x=radar_r[0]*10
      radar_y=radar_r[1]
      device_radar=np.array([[radar_x,radar_y-1],[radar_x,radar_y],[radar_x,radar_y+1]])
      valid_leng=np.clip(int(path_r[-1]),0,50)
      valid_point=path_r[:valid_leng]
      #print(y_steer_p[0][10:],valid_point[10:])
      valid_front=front[:valid_leng]
      z=np.zeros([len(valid_point)])
      #y_steer_p+=valid_point[0]
      #print(valid_point/y_steer_p[0])
      device_path=np.concatenate([valid_front.reshape(-1,1),valid_point.reshape(-1,1),z.reshape(-1,1)],axis=-1)
      #device_path=np.concatenate([front[:valid_leng].reshape(-1,1),-1*y_steer[:valid_leng].reshape(-1,1),z.reshape(-1,1)],axis=-1)
      print(front,y_steer_p,z,valid_leng)
      device_path_steer=np.concatenate([front[:valid_leng].reshape(-1,1),y_steer_p[0,:valid_leng].reshape(-1,1),z.reshape(-1,1)],axis=-1)
      print(y_steer_p,valid_point)
      #print(device_path[0],device_path_steer[0])
      #draw_path(y,x,(255,255,0),imgg,calibration,None,None)
      
      #thesis fig3.14
      draw_path1(device_path_steer,imgg,line_color=(0,0,0))
      #draw_path(device_radar[:,1],device_radar[:,0],(255,255,0),imgg,calibration,None)
      draw_path1(device_path,imgg,line_color=(255,0,0))
      
      
      #a=cv2.imread('tes.png')
      #print(a)
      
     
      rnn_state=model1.predict(x=[img_r,rnn_state])[0:1,106:]
      predictions_a=model.predict(x=[img_r])
      x1,y1=getvalidpoint(predictions_a)
      z1=np.zeros([len(x1)])
      device_path1=np.concatenate([x1.reshape(-1,1),y1.reshape(-1,1),z1.reshape(-1,1)],axis=-1)
      #draw_path(device_path1,random_img_raw,line_color=(0,255,0))
      #print(device_path,device_path1)
      predictions_r=model2.predict(x=[img_r])
      x2,y2=getvalidpoint(predictions_r)
      z2=np.zeros([len(x2)])
      device_path2=np.concatenate([x2.reshape(-1,1),y2.reshape(-1,1),z2.reshape(-1,1)],axis=-1)
      #draw_path(device_path2,random_img_raw,line_color=(0,0,255))
      predictions_gru=model1.predict(x=[img_r,rnn_state])
      x3,y3=getvalidpoint(predictions_gru)
      z3=np.zeros([len(x3)])
      device_path3=np.concatenate([x3.reshape(-1,1),y3.reshape(-1,1),z3.reshape(-1,1)],axis=-1)
      #draw_path(device_path3,random_img_raw,line_color=(255,255,0))
      #plotimg=cv2.resize(random_img_raw,(640,480))
      cv2.imwrite('./results/result%d_%d.png'%(num,num),imgg[:,:,::-1])
      
      #img_rr=cv2.resize(img_r[0],(640,480))
      #allimg.append(img_rr)
      #thesis fig3.13
      cv2.imwrite('./results/result%d_%d_%d.png'%(num,num,num),img_r[0])
      plt.figure()
      plt.plot(valid_point[:],x[:valid_leng],label='ground truth',color='r')
      #plt.plot(y1,x1,label='attention model',color='g')
      #plt.plot(y2,x2,label='reg model',color='b')
      #plt.plot(y3,x3,label='gru model',color='y')
      plt.plot(y_steer_p[0][:valid_leng],front[:valid_leng],color='k',label='steering trajectory')
      plt.legend()
      #plt.xlim(-2,2)
      plt.xlabel('lateral distance')
      plt.ylabel('front distance')
      plt.savefig('./results/result%d.png'%(num))
      #pltimg=cv2.imread('./results/result%d.png',(num))
      
      #result=np.concatenate([img_rr,pltimg],axis=1)
      #imgname='result%d.png'%num
      #cv2.imwrite('./results/'+imgname,result)
      num+=1
  #imgg1=np.concatenate(img1,axis=0)
  #cv2.imwrite('exampleimg1.png',imgg1)
  #imgg=np.concatenate(allimg,axis=0)
  #imgg2=np.concatenate(img2,axis=0)
  #imgg3=np.concatenate([imgg,imgg1,imgg2],axis=1)
  
  #cv2.imwrite('exampltimg.png',imgg3)
