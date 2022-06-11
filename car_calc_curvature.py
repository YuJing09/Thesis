import numpy as np
import matplotlib.pyplot as plt


def calc_curvature_rav4(v_ego,angle_steers,angle_offset=0):
  deg_to_rad=np.pi/180.
  slip_factor=0.000001
  steer_ratio=16.88
  wheel_base=2.65
  angle_steers_rad=(angle_steers-angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio*wheel_base * (1.+slip_factor*v_ego**2))
  return curvature
def calc_lookahead_offset_rav4(v_ego,angle_steers,d_lookahead,angle_offset=0):
  curvature = calc_curvature_rav4(v_ego,angle_steers,angle_offset)
  y_actual= d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature,-0.999,0.999))/2.)
  return y_actual 

def calc_curvature_civic(v_ego,angle_steers,angle_offset=0):
  deg_to_rad=np.pi/180.
  slip_factor=0.0014
  steer_ratio=15.38
  #steer_ratio=1
  wheel_base=2.7
  angle_steers_rad=(angle_steers-angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio*wheel_base * (1.+slip_factor*v_ego**2))
  return curvature

def calc_lookahead_offset_civic(v_ego,angle_steers,d_lookahead,angle_offset=0):
  curvature = calc_curvature_civic(v_ego,angle_steers,angle_offset)
  #print(curvature)
  y_actual= d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature,-0.999,0.999))/2.)
  return y_actual
if __name__=="__main__":

  a=5
#x_path=np.arange(0.,50.1,0.5)

