import numpy as np
from modell import EfficientNet,BiFPN,SeparableConvBlock
import tensorflow as tf

from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
import itertools
from loss import FocalLoss
from detbackbone import  EfficientDetBackbone
from server import client_generator
import argparse,os
def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y= tup
    
    
    Y= Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    w=tf.convert_to_tensor(Y)
    
    yield X,Y

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=30, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # 使用编号为1，2号的GPU
  model=EfficientDetBackbone(0)
  input1=tf.keras.Input(shape=(512,512,3))
  output=model(input1)
  
  model =tf.keras.Model(inputs=[input1],outputs=[output])
  model.summary()
  lossf=FocalLoss()
  g=gen(20, args.host, port=args.port)
  x,y=next(g)
  model.compile(optimizer="adam",loss=lossf.call)
  model.fit(x,y, batch_size=1)
