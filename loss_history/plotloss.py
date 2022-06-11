import numpy as np
import matplotlib.pyplot as plt
loss=np.load('loss.npy')
val_loss=np.load('val_loss.npy')
epoch=np.arange(200)+1
plt.plot(epoch,loss,color='r',label='train_loss')
plt.plot(epoch[10:],val_loss[10:],color='b',label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
