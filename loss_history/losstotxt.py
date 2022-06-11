import numpy as np

loss_name=input('loss_name:')
val_loss_name=input('val_loss_name:')
try:
  with open(loss_name+'.txt','a') as f:
    loss=np.load(loss_name+'1'+'.npy')
    for l in loss:
      f.write(str(l)+'\n')
  with open(val_loss_name+'.txt','a') as f:
    val_loss=np.load(val_loss_name+'1'+'.npy')
    for l in val_loss:
      f.write(str(l)+'\n')
except:
  print('Error file name')
