import lib.orientation as orient
import lib.coordinates as coord
from lib.camera import img_from_device, denormalize, view_frame_from_device_frame
import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
path='./14/'
frame_times=np.load(path+'global_pose/frame_times')
frame_positions=np.load(path+'global_pose/frame_positions')
frame_orientations=np.load(path+'global_pose/frame_orientations')
frame_velocities=np.load(path+'global_pose/frame_velocities')
v=np.load(path+'processed_log/CAN/speed/value')
r=np.sum(frame_velocities**2,axis=-1)
r=r**0.5
ll=r*0.05
#ecef_from_local=orient.rot_from_quat(frame_orientations[0])
#local_from_ecef=ecef_from_local.T
theta=[]
p=[[0,0]]
for i in range(len(r)):
  ecef_from_local=orient.rot_from_quat(frame_orientations[i])
  local_from_ecef=ecef_from_local.T
  frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[i:] - frame_positions[i])
  #print(len(v))
  if i < len(r)-1:
    vector=frame_positions_local[1]
    theta.append(np.arctan(vector[1]/vector[0]))
    p.append([ll[i]*np.cos(theta[-1]),ll[i]*np.sin(theta[-1])])
    print(ll[i])
p=np.array(p)
print(p)
    #print(vector,theta[i])
x=[np.sum(p[:i,0]) for i in range(len(r))]
y=[np.sum(p[:i,1]) for i in range(len(r))]
frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[0])
#theta=[np.arctan(i[1]/i[0]) for i in frame_positions_local]
#print(theta)

#x=frame_positions_local[:,0]
#y=frame_positions_local[:,1]

plt.plot(y,x)
plt.savefig('path2.png')

