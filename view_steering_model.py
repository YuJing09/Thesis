#!/usr/bin/env python
import argparse
import sys
import numpy as np
import h5py
import pygame
import json
import cv2

from keras.models import model_from_json
from matplotlib import pyplot as plt
from tools.lib.logreader import LogReader
from tools.lib.framereader import FrameReader
from common.transformations.camera import FULL_FRAME_SIZE, eon_intrinsics
from common.transformations.model import (MODEL_CX, MODEL_CY, MODEL_INPUT_SIZE,
                                          get_camera_frame_from_model_frame)
from tools.replay.lib.ui_helpers import (draw_lead_car, draw_lead_on, draw_mpc,
                                         draw_path, draw_steer_path,
                                         init_plots, to_lid_pt, warp_points)
from sklearn.metrics import r2_score

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
pygame.init()
pygame.font.init()
size = (640+640,480+160)
pygame.display.set_caption("comma.ai data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.fill((64,64,64))

camera_surface_bb = pygame.surface.Surface((640,480),0,24).convert()
camera_surface_w = pygame.surface.Surface((320,160),0,24).convert()
camera_surface_t= pygame.surface.Surface((320,160),0,24).convert()
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
    return _FULL_FRAME_TO_BB.dot(self._K).dot(self._E[:, [0, 1, 3]])


# ***** get perspective transform for images *****




def init_plots(arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles, bigplots=True):
  import matplotlib
  color_palette = { "r": (1,0,0),
                    "g": (0,1,0),
                    "b": (0,0,1),
                    "k": (0,0,0),
                    "y": (1,1,0),
                    "p": (0,1,1),
                    "m": (1,0,1) }

  if bigplots == True:
    fig = plt.figure(figsize=(6.4,6.4))
  elif bigplots == False:
    fig = plt.figure()
  else:
    fig = plt.figure(figsize=bigplots)

  fig.set_facecolor((0.2,0.2,0.2))

  axs = []
  for pn in range(len(plot_ylims)):
    ax = fig.add_subplot(len(plot_ylims),1,len(axs)+1)
    ax.set_xlim(plot_xlims[pn][0], plot_xlims[pn][1])
    ax.set_ylim(plot_ylims[pn][0], plot_ylims[pn][1])
    ax.patch.set_facecolor((0.4, 0.4, 0.4))
    axs.append(ax)

  plots = [] ;idxs = [] ;plot_select = []
  for i, pl_list in enumerate(plot_names):
    for j, item in enumerate(pl_list):
      plot, = axs[i].plot(arr[:, name_to_arr_idx[item]],
                          label=item,
                          color=color_palette[plot_colors[i][j]],
                          linestyle=plot_styles[i][j])
      
      plots.append(plot)
      idxs.append(name_to_arr_idx[item])
      plot_select.append(i)
    axs[i].set_title(", ".join("%s (%s)" % (nm, cl)
                               for (nm, cl) in zip(pl_list, plot_colors[i])), fontsize=10)
    if i < len(plot_ylims) - 1:
      axs[i].set_xticks([])#last figue show xtick

  
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()

  

  def draw_plots(arr):
    for ax in axs:
      
      ax.draw_artist(ax.patch)
    for i in range(len(plots)):
      
      plots[i].set_ydata(arr[:, idxs[i]])
      axs[plot_select[i]].draw_artist(plots[i])

    

    raw_data = renderer.tostring_rgb()
    
    plot_surface = pygame.image.frombuffer(raw_data, fig.canvas.get_width_height(), "RGB").convert()
    return plot_surface
  return draw_plots





# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.005 # slip factor obtained from real data
  steer_ratio = 16.88   # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.65  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature



# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Path viewer')
  parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
  parser.add_argument('--dataset', type=str, default="2016-01-30--11-24-51", help='Dataset/video clip name')
  
  args = parser.parse_args()
  
  with open('./outputs/steering_model/'+args.model, 'r') as jfile:
    model = model_from_json(json.load(jfile))

  model.compile("sgd", "mse")
  weights_file = './outputs/steering_model/'+args.model.replace('json', 'keras')
  model.load_weights(weights_file)
  
  # default dataset is the validation data on the highway
  dataset = args.dataset
  
  path='//mnt/usb//Chunk_1//b0c9d2329ad1606b_2018-07-30--13-44-30//11//'
  log = h5py.File(path+"log.h5", "r")
  model.save('test.h5')
  
  
  
  
  
  
  
  
  
  
  
  
  name_to_arr_idx = { 
                      "angle_steers": 0,
                      "predicted_steers":1,
                      "speed":2,
                      "loss":3
                      
                      
                     }
  plot_arr=np.zeros([100,len(name_to_arr_idx.values())])
  
  plot_xlims=[[0,100],[0,100],[0,100]]
  plot_ylims=[[-50,50],[0,50],[0,200]]
  plot_names=[['angle_steers','predicted_steers'],['speed'],['loss']]
  plot_colors=[['b','r'],['r'],['r']]
  plot_styles=[['-','-'],['-'],['-']]
 
  intrinsic_matrix=eon_intrinsics
  
  
  
  draw_plots = init_plots(plot_arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles, bigplots=True)
  # skip to highway
  g=0
  
  test_loss=[]
  pre=[]
  pause=False
  font=pygame.font.Font(None,30)
  fr=FrameReader(path+'video.hevc')
  #frame=np.array(fr.get(0,count=fr.frame_count,pix_fmt='rgb24'))
  while True and g<fr.frame_count:
    for event in pygame.event.get():
         if event.type==pygame.QUIT:
            sys.exit()
         if event.type==pygame.KEYDOWN:
            if event.key ==pygame.K_p:
               pause=not pause
          
    if not pause:
       i=g%fr.frame_count
       extrinsic_matrix=log['extrinsicmatrix'][i]
    
       ke = intrinsic_matrix.dot(extrinsic_matrix)
       warp_matrix = get_camera_frame_from_model_frame(ke)
    
       calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)
       cpw = warp_points(CalP, calibration.model_to_bb())
       vanishing_pointw = warp_points(vanishing_point, calibration.model_to_bb())
       frame=np.array(fr.get(i,count=1,pix_fmt='rgb24'))
       print cpw,cpw[1][0]-cpw[0][0],cpw[2][1]-cpw[1][1]
       img=frame[0]
       img=cv2.warpAffine(img, (_BB_TO_FULL_FRAME)[:2],
        		(640, 480), dst=img, flags=cv2.WARP_INVERSE_MAP)
       
    
       imgw = cv2.warpAffine(img,(calibration.model_to_bb())[:2], (320,160), flags=cv2.WARP_INVERSE_MAP)
    
       imgt=imgw.copy()
    
    
       plot_arr[1:]=plot_arr[:-1]
       t=imgw.swapaxes(0,1)
       tt=t.swapaxes(0,2) 
    
    
    
    
    
    
       predicted_angle = model.predict(imgw[None, :, :, :])[0][0]
    
       pre.append(predicted_angle)
    
       angle_steers = log['steering_angle'][i]
       predicted_steers=0
       speed_ms = log['speed'][i]
       test_loss=((predicted_angle-angle_steers)**2)
    
       s_angle='steering_angle: %f'% (angle_steers)
       s_predicted_angle='predicted_angle: %f'%(predicted_angle)
       s_speed='speed_ms: %f'%(speed_ms)
       s_loss='loss: %f'%(test_loss) 
       s_vanishing_point='vp:%f,%f'%(vanishing_point[0][0],vanishing_point[0][1])
       plot_arr[0,name_to_arr_idx['angle_steers']]=angle_steers
       plot_arr[0,name_to_arr_idx['predicted_steers']]=predicted_angle
       plot_arr[0,name_to_arr_idx['speed']]=speed_ms
       plot_arr[0,name_to_arr_idx['loss']]=test_loss
 
       curvature=calc_curvature(speed_ms,angle_steers)
    
       curvature2=calc_curvature(speed_ms,predicted_angle)
    
       curvature1=log['curvature'][i]
       predicted_curvature=0
    
       VM=None
       draw_steer_path(speed_ms, curvature, BLUE, imgw, calibration, None, None, BLUE)
       draw_steer_path(speed_ms, curvature2, RED, imgw, calibration, None, None, BLUE)
    
    
       pygame.surfarray.blit_array(camera_surface_bb,img.swapaxes(0,1))
       pygame.surfarray.blit_array(camera_surface_w, imgw.swapaxes(0,1))
       pygame.surfarray.blit_array(camera_surface_t, imgt.swapaxes(0,1))
    
       
       screen.blit(camera_surface_bb, (0,0))
       screen.blit(camera_surface_t,(0,480))
       screen.blit(camera_surface_w,(320,480))
       screen.blit(draw_plots(plot_arr),(640,0))
       screen.blit(font.render(s_angle,True,(0,255,0)),(0,font.get_linesize()*0))
       screen.blit(font.render(s_predicted_angle,True,(0,255,0)),(0,font.get_linesize()*1))
       screen.blit(font.render(s_speed,True,(0,255,0)),(0,font.get_linesize()*3))
       screen.blit(font.render(s_loss,True,(0,255,0)),(0,font.get_linesize()*2))
       screen.blit(font.render(s_vanishing_point,True,(0,255,0)),(0,font.get_linesize()*4))
       pygame.draw.polygon(screen, BLUE, tuple(map(tuple, cpw)), 1)
       pygame.draw.circle(screen, BLUE, map(int, map(round, vanishing_pointw[0])), 2)
       pygame.display.flip()
       g+=1
print r2_score(log['steering_angle'],pre)
