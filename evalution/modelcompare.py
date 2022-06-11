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
from car_calc_curvature import calc_lookahead_offset_civic
os.environ['CUDA_VISIBLE_DEVICES'] = " "
def vandermondepoints(predict,l):
  vandermonde=tf.range(0,50,1,dtype=tf.float32)
  
  p=predict[:,:50]
  #l_v=np.clip(((predict[:,50]*10.).astype(np.int32)),0,50)
  w=predict[:,51:101]
  vandermonde2=tf.square(vandermonde)
  vandermonde3=tf.math.pow(vandermonde,3)
  v1=tf.reshape(vandermonde,[-1,1])
  v2=tf.reshape(vandermonde2,[-1,1])
  v3=tf.reshape(vandermonde3,[-1,1])
  p_vs=[]
  vandermonde=tf.concat([v1,v2,v3],axis=-1)
  batch_size=tf.shape(p)[0]
  w=tf.math.exp(w)
  
  for i in range(batch_size):
    #l=int(l_v[i])
    
    point=p[i,:l]
    point=point-point[0]
    weight=w[i,:l]
    weight=tf.reshape(weight,(-1,1))**0.001
    #print(weight)
    point=tf.reshape(point,(-1,1))
    y=point/weight
    
    x=vandermonde[:l]/weight
    #print(x,y)
    vander=tf.linalg.pinv(x)
    
    pv=tf.matmul(vander,y)
    
    y_p_v=tf.matmul(vandermonde[:l],pv)[:,-1]+p[i,0]
    #print(y_p_v,p)
    p_vs.append(y_p_v)
  return p_vs
def custom_loss(y_true, y_pred,delta=2.):
  batch_size=tf.shape(y_true)[0]
  valid_len=y_true[:,50]*10.
  valid_len=tf.reshape(valid_len,[-1,1])
  true_valid_len=y_true[:,50]
  pred_valid_len=y_pred[:,50]
  
  vandermonde=tf.range(0,50,1,dtype=tf.float32)
 
  vandermonde2=tf.square(vandermonde)
  vandermonde3=tf.math.pow(vandermonde,3)
  v1=tf.reshape(vandermonde,[-1,1])
  v2=tf.reshape(vandermonde2,[-1,1])
  v3=tf.reshape(vandermonde3,[-1,1])
  
  vandermonde=tf.concat([v1,v2,v3],axis=-1)
  weights=tf.nn.softmax(y_pred[:,51:101],axis=-1)
  #valid_len_loss=tf.reduce_sum(y_true[:,51]-y_pred[:,101])/tf.cast(batch_size,dtype=tf.float32)
  true_points=y_true[:,:50]
  pred_points=y_pred[:,:50]
  true_lead=y_true[:,51:55]
  pred_lead=y_pred[:,101:105]
  true_lead_prob=y_true[:,55]
  pred_lead_prob=tf.clip_by_value(tf.math.sigmoid(y_pred[:,105]),1e-15,1-1e-15)
  s=0.
  loss=tf.zeros([4],tf.float32)
  for i in range(batch_size):
    
    l=tf.cast(valid_len[i][0],dtype=tf.int32)
   
    l=tf.clip_by_value(l,0,50)
    error_lead=tf.abs(tf.subtract(true_lead[i,:],pred_lead[i,:]))
   
    
    condition_lead=tf.less(error_lead,delta)
    

    
    small_res_lead=0.5 * tf.square(error_lead)
    
   
    
    large_res_lead= delta * error_lead - 0.5 * tf.square(delta) 
  
    
    
    lead_loss=tf.where(condition_lead,small_res_lead,large_res_lead)
   
    
    #lead loss
    lead_all_loss=tf.where(tf.equal(true_lead_prob[i],1),tf.reduce_sum(lead_loss),0)
    
    
    point=pred_points[i,:]
    point=point-point[0]
    weight=weights[i,:]
    weight=tf.reshape(weight,(-1,1))
    point=tf.reshape(point,(-1,1))
    y=point*weight
    #y=y[:l]
    
    x=vandermonde[:]*weight
    #x=x[:l]
    vander=tf.linalg.pinv(x)
    p=tf.matmul(vander,y)
    y_p = y_pred[i,:50]
    y_p_v=tf.matmul(vandermonde,p)[:,-1]+pred_points[i,0]
    y_t=y_true[i,:50]
    #print(y_p[:l],y_p_v[:l],y_t[:l],weight)
    #huber loss 
 
    error_points= tf.abs(tf.subtract(y_p,y_t))
    error_points_v= tf.abs(tf.subtract(y_p_v,y_t))
    
    condition_points=tf.less(error_points,delta)
    condition_points_v=tf.less(error_points_v,delta)
    
    small_res_points=0.5 * tf.square(error_points)  #mse
    small_res_points_v=0.5 * tf.square(error_points_v)  #mse
    
    large_res_points= delta * error_points - 0.5 * tf.square(delta)
    large_res_points_v= delta * error_points_v - 0.5 * tf.square(delta)
    points_loss=tf.where(condition_points,small_res_points,large_res_points)
    
    points_v_loss=tf.where(condition_points_v,small_res_points_v,large_res_points_v)  
    
    
    loss_v=tf.reduce_sum(points_v_loss[:l])
    loss_p=tf.reduce_sum(points_loss[:l])

    error_valid_len=tf.abs(tf.subtract(true_valid_len[i],pred_valid_len[i]))
    condition_valid_len=tf.less(error_valid_len,delta)
    small_res_valid_len=0.5 * tf.square(error_valid_len)  #mse
    large_res_valid_len=delta * error_valid_len - 0.5 * tf.square(delta)
    valid_len_loss=tf.where(condition_valid_len,small_res_valid_len,large_res_valid_len)
    true_prob=true_lead_prob[i]
    pred_prob=pred_lead_prob[i]
    bce=0.5 * tf.square(true_prob-pred_prob)
    
    error_prob=tf.abs(tf.subtract(pred_prob,true_prob))
    #print(true_prob,pred_prob)
    #bce=tf.where(tf.less(error_prob,0.8),bce,0.)
    s=s+valid_len_loss+loss_p+lead_all_loss
    aa=tf.cast([loss_p,valid_len_loss,lead_all_loss,bce],dtype=tf.float32)
    loss+=aa
    
    
  all_loss=s/tf.cast(batch_size,dtype=tf.float32)
  loss=loss/tf.cast(batch_size,dtype=tf.float32)
  return all_loss,loss
def custom_loss2(y_true, y_pred,delta=2.):
  batch_size=tf.shape(y_true)[0]
  #valid_len=y_true[:,50]
  #valid_len=tf.reshape(valid_len,[-1,1])
  true_valid_len=tf.clip_by_value(y_true[:,50:51],0.,1000.)
  pred_valid_len=y_pred[:,50]
  true_std=tf.abs(y_true[:,:50]-y_pred[:,:50])
  pred_std=tf.math.exp(y_pred[:,51:101])
  
  #valid_len_loss=tf.reduce_sum(y_true[:,51]-y_pred[:,101])/tf.cast(batch_size,dtype=tf.float32)
  true_points=y_true[:,:50]
  pred_points=y_pred[:,:50]
  true_lead=y_true[:,51:55]
  
  true_lead_prob=y_true[:,55]
  pred_lead=y_pred[:,101:105]
  loss_res=tf.zeros([4],tf.float32)
  #print(true_lead,pred_lead)
  #true_lead_std=tf.abs(true_lead-pred_lead)
 
  #pred_lead_std=tf.math.exp(y_pred[:,105:109])
  #pred_lead_prob=tf.clip_by_value(tf.math.sigmoid(y_pred[:,105]),1e-15,1-1e-15)
  #lead_prob_mse=tf.square(pred_lead_prob-true_lead_prob)/2.
  #print(true_lead_prob,pred_lead_prob)  
  li=tf.cast(tf.clip_by_value(y_true[:,50]*10.,0.,50.),dtype=tf.float32)
 
 

  l=tf.cast(true_valid_len*10.,dtype=tf.int32)
  l=tf.clip_by_value(l,0,50)
  ll=tf.broadcast_to(l,(batch_size,50))
  len_x=tf.range(0,50,1,dtype=tf.int32)
  cond_len=tf.less(len_x,ll)
    #points huber loss 
  error_zero_point=tf.abs(pred_points[:,0])
  error_zero_point_std=tf.abs(tf.subtract(pred_points[:,0],pred_std[:,0]))
  error_point=tf.abs(tf.subtract(true_points[:,:],pred_points[:,:]))
  error_std=tf.abs(tf.subtract(true_std[:,:],pred_std[:,:]))
  error_valid_len=tf.abs(tf.subtract(true_valid_len[:,-1],pred_valid_len[:]))
  
  condition_zero_point=tf.less(error_zero_point,delta)
  condition_zero_point_std=tf.less(error_zero_point_std,delta)
  condition_point=tf.less(error_point,delta)
  condition_std=tf.less(error_std,delta)
  condition_valid_len=tf.less(error_valid_len,delta)
    
  small_res_point=0.5 * tf.square(error_point)  #mse
  small_res_std=0.5 * tf.square(error_std) #mse
  small_res_valid_len=0.5 * tf.square(error_valid_len)  #mse
  small_res_zero_point=0.5 * tf.square(error_zero_point) 
  small_res_zero_point_std=0.5 * tf.square(error_zero_point_std)
    
  large_res_point= delta * error_point - 0.5 * tf.square(delta) 
  large_res_std= delta * error_std - 0.5 * tf.square(delta) 
  large_res_valid_len=delta * error_valid_len - 0.5 * tf.square(delta)
  large_res_zero_point= delta * error_zero_point - 0.5 * tf.square(delta) 
  large_res_zero_point_std= delta * error_zero_point_std - 0.5 * tf.square(delta)
    
  point_loss=tf.where(condition_point,small_res_point,large_res_point)
  point_zero_point_loss=tf.where(condition_zero_point,small_res_zero_point,large_res_zero_point)
  point_zero_point_std_loss=tf.where(condition_zero_point_std,small_res_zero_point_std,large_res_zero_point_std)
  std_loss=tf.where(condition_std,small_res_std,large_res_std)
  valid_len_loss=tf.where(condition_valid_len,small_res_valid_len,large_res_valid_len)
  
  #point_loss=tf.reduce_sum(point_loss[:l])
 # std_loss=tf.reduce_sum(std_loss[:l])
  points_loss=tf.reduce_sum(tf.where(cond_len,point_loss,0),axis=1)
  stds_loss=tf.reduce_sum(tf.where(cond_len,std_loss,0),axis=1)
  
  #points loss
  points_all_loss=tf.where(tf.equal(li,0),point_zero_point_loss,(tf.math.divide(points_loss,li)))
  
  points_std_all_loss=tf.where(tf.equal(li,0),point_zero_point_std_loss,(tf.math.divide(stds_loss,li)))
  
  #lead huber loss
  error_lead=tf.abs(tf.subtract(true_lead[:,:],pred_lead[:,:]))
 # error_lead2=tf.abs(tf.subtract(true_lead[:],pred_lead[:]))
  
 
  #error_lead_std=tf.abs(tf.subtract(true_lead_std[:,:],pred_lead_std[:,:]))
    
  condition_lead=tf.less(error_lead,delta)
    
  #condition_lead_std=tf.less(error_lead_std,delta)
    
  small_res_lead=0.5 * tf.square(error_lead)
    
  #small_res_lead_std=0.5 * tf.square(error_lead_std) 
    
  large_res_lead= delta * error_lead - 0.5 * tf.square(delta) 
  
  #large_res_lead_std= delta * error_lead_std - 0.5 * tf.square(delta) 
    
  lead_loss=tf.where(condition_lead,small_res_lead,large_res_lead)
  #lead_std_loss=tf.where(condition_lead_std,small_res_lead_std,large_res_lead_std)
  
  #lead loss
  lead_all_loss=tf.where(tf.equal(true_lead_prob[:],1),tf.reduce_sum(lead_loss,axis=1),0)
  
  #lead_std_all_loss=tf.where(tf.equal(true_lead_prob[:],1),tf.reduce_sum(lead_std_loss,axis=1),0) 
  
    #bce
  true_prob=true_lead_prob[:]
  #pred_prob=pred_lead_prob[:]
  #bce=0.5 * tf.square(true_prob-pred_prob)
  #print(true_prob-pred_prob)
  #error_prob=tf.abs(tf.subtract(pred_lead_prob,true_lead_prob))
  #bce=tf.where(tf.less(error_prob,0.8),bce,0.)
  #print(points_all_loss,lead_all_loss,valid_len_loss)
    #bce=-1.*(true_prob*tf.math.log(pred_prob)+((1.-true_prob)*tf.math.log(1.-pred_prob)))
    #s add all loss
  #print(points_all_loss,points_std_all_loss,bce,lead_all_loss,lead_std_all_loss,valid_len_loss)
  
  #loss=points_all_loss+points_std_all_loss/1000.+lead_all_loss+lead_std_all_loss/1000.+valid_len_loss+bce
  loss=points_std_all_loss/1000.+points_all_loss+lead_all_loss+valid_len_loss
    #all_loss=points_all_loss
  
  aa=tf.cast([points_all_loss[0],points_std_all_loss[0],valid_len_loss[0],lead_all_loss[0]],dtype=tf.float32)
  #print(loss)  
    #
    #print(points_all_loss,points_std_all_loss,lead_all_loss,lead_std_all_loss,valid_len_loss)
  loss_res+=aa
  

  
  return loss,loss_res
def RMSE(x,y):
  y=np.array(y)
  
  
  return (np.sum(0.5*(x-y)**2))


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
  model3=Ypregsteer(2)
  output3=model3.call(input1)
  model3=tf.keras.Model(inputs=[input1],outputs=output3) 
  #model3.summary()
  #model3=Ypgru2(2)
  #output3=model3.call(input1,input2)
  #model3=tf.keras.Model(inputs=[input1,input2],outputs=output3)
  model.load_weights('./saved_model/weights3-improvement-02-2.01.hdf5')
  model1.load_weights('./saved_model/weights-gru-improvement-85-2.08.hdf5')
  model2.load_weights('./saved_model/weights-reg-improvement-30-2.02.hdf5')
  model3.load_weights('./saved_model/steer-weights-improvement-159-0.59.hdf5')
  data_len=len(path)
  test_path=path[-1*int(0.25*data_len):]
  print(len(test_path))
  loss=[]
  vandermonde=np.reshape(np.arange(50),[-1,1])
  vandermonde=np.concatenate([vandermonde,vandermonde**2,vandermonde**3],axis=-1)
  loss_l=4
  cc=np.zeros([loss_l],dtype=np.float32)
  dd=np.zeros([loss_l],dtype=np.float32)
  ee=np.zeros([loss_l],dtype=np.float32)
  ff=np.zeros([1],dtype=np.float32)
  gg=np.zeros([1],dtype=np.float32)
  r=0
  m=0
  mm=0
  ade=np.zeros([4])
  fde=np.zeros([4])
  radar_all=np.zeros([3,4])
  for i in test_path[:]:
    print(i)
    with h5py.File(i,'r') as c:
      try:
        speed_path=i.replace('camera320.h5','global_pose/frame_velocities')
        steer_path=i.replace('camera320.h5','global_pose/frame_steers.npy')
        path_data=i.replace('camera320.h5','pathdata.h5')
        radarpath=i.replace('camera320','radardata')
      
        speed=np.load(speed_path)
        steer=np.load(steer_path)
        print(len(steer))
        path=h5py.File(path_data,'r')
        l5=h5py.File(radarpath,'r')
        P=path['Path']
        img=c['X']
      except:
        continue
      leadOne=l5['LeadOne'][:]
      #if not np.sum(leadOne[:,-1])==len(leadOne):
      #  continue
      #m+=len(P)
      #mm+=np.sum(leadOne[:,-1])
      #print(m,mm)
      #continue
      rnn_state0=np.zeros([1,512])
      rnn_state1=np.zeros([1,512])
      rnn_state=np.zeros([1,512])
      l1=[]
      l2=[]
      l3=[]
      l4=[]
      front=np.arange(50)+1
      a=np.zeros([loss_l],dtype=np.float32)
      b=np.zeros([loss_l],dtype=np.float32)
      d=np.zeros([loss_l],dtype=np.float32)
      loss_v=[]
      loss_p=[]
      #e=np.zeros([5],dtype=np.float32)
      out=np.concatenate((P,leadOne),axis=-1)
      out[:,50:52]/=10.
      c0=[]
      c1=[]
      c2=[]
      c3=[]
      c4=[]
      for j in range(0,len(img),5):
        
        #print(rnn_state.shape,img[j:j+1].shape)
        if np.clip(int(out[j,50]*10.),0,50) !=0:
          predictions=model.predict(x=[img[j:j+1]])
          pv=vandermondepoints(predictions,np.clip(int(out[j,50]*10.),0,50))
          p=predictions[0,:np.clip(int(out[j,50]*10.),0,50)]
          true_p=out[j,:np.clip(int(out[j,50]*10.),0,50)]
          #print(pv,p)
        #print(p,true_p,RMSE(p,true_p))
          #print(RMSE(pv[-1].numpy(),true_p),RMSE(true_p,p))
          loss_v.append(RMSE(pv[-1].numpy(),true_p))
          loss_p.append(RMSE(true_p,p))
        #print(pv,p,true_p)
        #rnn_state=predictions[0:1,106:]
        steer_p=-1*model3.predict(x=[img[j:j+1]])
        speed_g=speed[j]
        steer_g=steer[j]
        speed_g=np.linalg.norm(speed_g)
        ground_truth_p=out[j][:50]
        l=np.clip(int(out[j][50]*10),0,50)
        len_g=out[j][50]*10
        rl_g=out[j][51]*10
        ry_g=out[j][52]
        rv_g=out[j][53]
        ra_g=out[j][54]
        #print(l)
        attention_point=predictions[0,:50]
        at_l=predictions[0][50]*10
        
        at_rl=predictions[0][101]*10
        at_ry=predictions[0][102]
        at_rv=predictions[0][103]
        at_ra=predictions[0][104]
        gr_ra=np.array([rl_g,ry_g,rv_g,ra_g])
        #print(len_g,at_l,rl_g,at_rl,ry_g,at_ry,rv_g,at_rv,ra_g,at_ra)
        ade1=np.sum((ground_truth_p[:l]-attention_point[:l])**2)/l
        fde1=np.abs((ground_truth_p[l-1]-attention_point[l-1]))
        loss0,loss0_resp=custom_loss2(out[j:j+1],predictions)
        a+=loss0_resp.numpy()
        aa=a.copy()
        predictions1=model1.predict(x=[img[j:j+1],rnn_state])
        gru_l=predictions1[0][50]*10
        gru_rl=predictions1[0][101]*10
        gru_ry=predictions1[0][102]
        gru_rv=predictions1[0][103]
        gru_ra=predictions1[0][104]
        gru_point=predictions1[0,:50]
        ade2=np.sum((ground_truth_p[:l]-gru_point[:l])**2)/l
        fde2=np.abs((ground_truth_p[l-1]-gru_point[l-1]))
        rnn_state=predictions1[0:1,106:]
        loss1,loss1_resp=custom_loss2(out[j:j+1],predictions1)
        predictions2=model2.predict(x=[img[j:j+1]])
        reg_l=predictions2[0][50]*10
        
        reg_rl=predictions2[0][101]*10
        reg_ry=predictions2[0][102]
        reg_rv=predictions2[0][103]
        reg_ra=predictions2[0][104]
        #print(gr_ra)
        radar_p=out[j][-1]
        
        if radar_p:
          print(reg_rl,reg_ry,reg_rv,reg_ra,gr_ra)
          radar_all[0]+=0.5*(np.array([reg_rl,reg_ry,reg_rv,reg_ra])-gr_ra)**2
          radar_all[1]+=0.5*(np.array([gru_rl,gru_ry,gru_rv,gru_ra])-gr_ra)**2
          radar_all[2]+=0.5*(np.array([at_rl,at_ry,at_rv,at_ra])-gr_ra)**2
          r+=1
        else:
          continue
        #print(r)
        reg_point=predictions2[0,:50]
        c0.append([at_l,gru_l,reg_l])
        c1.append([at_rl,gru_rl,reg_rl])
        c2.append([at_ry,gru_ry,reg_ry])
        c3.append([at_rv,gru_rv,reg_rv])
        c4.append([at_ra,gru_ra,reg_ra])
        ade3=np.sum((ground_truth_p[:l]-reg_point[:l])**2)/l
        fde3=np.abs(ground_truth_p[l-1]-reg_point[l-1])
        np.sum((ground_truth_p[:l]-reg_point[:l]))
        rnn_state0=predictions2[0:1,106:]
        loss2,loss2_resp=custom_loss2(out[j:j+1],predictions2)
        #predictions3=model3.predict(x=[img[j:j+1],rnn_state])
        #rnn_state=predictions3[0:1,106:]
        #loss3,loss3_resp=custom_loss(out[j:j+1],predictions3)
        predictions_steer=model3.predict(x=[img[j:j+1]])
        #print(speed_g,steer_p)
        
        y_steer=calc_lookahead_offset_civic(speed_g,steer_p,front)
        y_steer_g=calc_lookahead_offset_civic(speed_g,steer_g,front)
        #print(y_steer[0])
        ade4=np.sum((ground_truth_p[:l]-y_steer[0][:l]))/l
        fde4=np.abs(ground_truth_p[l-1]-y_steer[0][l-1])
        #print(out[j][:10],steer_g)
        #print(y_steer)
        """if l !=0:
          ade+=np.array([ade1,ade2,ade3,ade4])
          fde+=np.array([fde1,fde2,fde3,fde4])
          r+=1
        else:
          continue"""
        b+=loss1_resp.numpy()
        bb=b.copy()
        d+=loss2_resp.numpy()
        #e+=loss3_resp.numpy()
        l1.append(loss0)
        l2.append(loss1)      
        l3.append(loss2)
        #l4.append(loss3)
        """predictions=model.predict(x=[img[j:j+1],rnn_state0])
        points=predictions[0,:50]
        rnn_state=predictions[0:1,110:]
        valid_len=np.clip(int(P[j][-1]),0,50)
        points=points[:valid_len]
        true_points=P[j][:valid_len]
        loss.append(RMSE(points,true_points))
        #print(loss[-1])
        predictions1=model1.predict(x=[img[j:j+1],rnn_state1])
        points1=predictions1[0,:50,np.newaxis]-predictions1[0,0,np.newaxis]
        error=np.exp(predictions1[0,50:100,np.newaxis])
        rnn_state1=predictions1[0:1,110:]
        vander=vandermonde[:valid_len]
        #print(vander.shape,error.shape)
        
        vander2=vander/error[:valid_len]
        #print(error[:valid_len],vander,vander2)
        v_p=np.linalg.pinv(vander2)
        points2=points1[:valid_len]/error[:valid_len]
        #print(v_p.shape,points2.shape)
        p=np.dot(v_p,points2)
        y_p=np.dot(vandermonde[:valid_len],p)+predictions1[0,0]
        y_p=y_p[:,0]
        #print(predictions1[0,0])
        a=RMSE(predictions1[0,:valid_len],true_points)
        b=RMSE(y_p,true_points)
        c=RMSE(points,true_points)
        #print(a,b,c)
        l1.append(a)
        l2.append(b)
        l3.append(c)"""
      c0=np.array(c0)
      c1=np.array(c1)
      c2=np.array(c2)
      c3=np.array(c3)
      c4=np.array(c4)
      frame_num=np.arange(len(c0))
      """
      plt.figure()
      #print(out[:100,50])
      #plt.plot(frame_num,out[:,50]*10/out[:,50]*10,linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c0[:,0]-out[:,50]*10),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c0[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c0[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotlen.png')
      
      plt.figure()
      #print(out[:100,50])
      plt.plot(frame_num,out[:,50]*10,linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c0[:,0]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c0[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c0[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotlen1.png')
      plt.figure()
      plt.plot(frame_num,out[:,51]*10,linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c1[:,0]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c1[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c1[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotrl1.png')
      plt.figure()
      #print(out[:100,50])
      #plt.plot(frame_num,out[:,51]*10,linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c1[:,0]-out[:,51]*10),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c1[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c1[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotrl.png')
      plt.figure()
      #print(out[:100,50])
      #plt.plot(frame_num,out[:,52],linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c2[:,0]-out[:,52]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c2[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c2[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotry.png')
      plt.figure()
      #print(out[:100,50])
      plt.plot(frame_num,out[:,52],linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c2[:,0]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c2[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c2[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotry1.png')
      plt.figure()
      #print(out[:100,50])
      plt.plot(frame_num,out[:,53],linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c3[:,0]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c3[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c3[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotrv1.png')
      plt.figure()
      #print(out[:100,50])
      #plt.plot(frame_num,out[:,53],linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c3[:,0]-out[:,53]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c3[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c3[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotrv.png')
      plt.figure()
      #print(out[:100,50])
      #plt.plot(frame_num,out[:,54],linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c4[:,0]-out[:,54]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c4[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c4[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotra.png')
      plt.figure()
      #print(out[:100,50])
      plt.plot(frame_num,out[:,54],linewidth=1,color='r',label='ground truth')
      plt.plot(frame_num,(c4[:,0]),linewidth=1,color='b',label='att')
      #plt.plot(frame_num,c4[:,1],linewidth=1,color='y',label='gru')
      #plt.plot(frame_num,c4[:,2],linewidth=1,color='b',label='reg')
      plt.legend()
      plt.savefig('plotra1.png')
      """
      #if len(loss_p)==0:
      #  loss_p.append(0)
      #  loss_v.append(0)
      #break
      cc=cc+(a/len(l1))
      dd=dd+(b/len(l2))
      ee=ee+(d/len(l3))
      #ff=ff+(np.sum(loss_v)/len(loss_v))
      #gg+=(np.sum(loss_p)/len(loss_p))
      #print(np.sum(l1)/len(l1),np.sum(l2)/len(l2),np.sum(l3)/len(l3),'-----',a/len(l1),b/len(l2),d/len(l3))
    print(radar_all/r)
    #print(cc/r,dd/r,ee/r)
    #print(ade/r,fde/r)
    path.close()
    l5.close()
    

        
