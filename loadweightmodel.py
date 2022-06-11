import tensorflow as tf
import argparse
import json 
import os
import cv2
import numpy as np
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from modell import EfficientNet
from server import client_generator

if __name__=="__main__":

  block_args,global_params=get_model_params('efficientnet-b2',None)
  model=EfficientNet(block_args,global_params)
  
  x=tf.keras.Input(shape=(260,260,3))
  output=model.call(x)
  
  model =tf.keras.Model(inputs=[x],outputs=output)
  model.summary()
  model.load_weights('./saved_model/eff2.keras')

