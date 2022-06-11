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
#from loss import FocalLoss
class Regressor(tf.keras.layers.Layer):
  def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5):
    super(Regressor, self).__init__()
    self.num_layers = num_layers
    self.in_channels=in_channels
    self.num_anchors=num_anchors
    self.pyramid_levels=pyramid_levels
    self._build()
  def _build(self):
    self.conv_list=[SeparableConvBlock(self.in_channels,self.in_channels,norm=None,activation=False) for _ in range(self.num_layers) ]
    self.bn_list =[[tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3) for i in range(self.num_layers)] for j in range(self.pyramid_levels)]
    self.header = SeparableConvBlock(self.in_channels, self.num_anchors * 4, norm=False, activation=False)
    self._relu_fn=tf.keras.layers.ELU()
  def call(self,inputs):
    feats = []
    for feat, bn_list in zip(inputs, self.bn_list):
      for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
        feat = conv(feat)
        feat = bn(feat)
        feat = self._relu_fn(feat)
      feat = self.header(feat)
      
      feat = tf.keras.layers.Reshape((-1,4))(feat)
      
      feats.append(feat)
    
    feats=tf.keras.layers.concatenate(feats,axis=1)
    return feats
class Classifier(tf.keras.layers.Layer):
  def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5):
    super(Classifier, self).__init__()
    self.num_anchors = num_anchors
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.in_channels = in_channels
    self.pyramid_levels=pyramid_levels
    self._build()
  def _build(self):
    self.conv_list =[SeparableConvBlock(self.in_channels, self.in_channels, norm=False, activation=False) for i in range(self.num_layers)]
    self.bn_list = [[tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3) for i in range(self.num_layers)] for j in range(self.pyramid_levels)]
    self.header = SeparableConvBlock(self.in_channels, self.num_anchors * self.num_classes, norm=False, activation=False)
    self._relu_fn=tf.keras.layers.ELU()
  def call(self,inputs):
    feats = []
    for feat, bn_list in zip(inputs, self.bn_list):
      for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
        feat = conv(feat)
        feat = bn(feat)
        feat = self._relu_fn(feat)
      feat = self.header(feat)
      feat = tf.keras.layers.Reshape((-1,self.num_classes))(feat)
      feats.append(feat)
    feats=tf.keras.layers.concatenate(feats,axis=1)
    feats=tf.keras.activations.sigmoid(feats)
    return feats
class Anchors():
  def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
    super().__init__()
    self.anchor_scale = anchor_scale

    if pyramid_levels is None:
        self.pyramid_levels = [3, 4, 5, 6, 7]
    else:
        self.pyramid_levels = pyramid_levels

    self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
    self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
    self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
    self.last_anchors = {}
    self.last_shape = None
  def get_boxes(self,image_shape):
    boxes_all = []
    for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)
    
    #anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
    return anchor_boxes
    
class EfficientDetBackbone(tf.keras.Model):
  def __init__ (self,compound_coef,num_classes=80,**kwargs):
    super(EfficientDetBackbone, self).__init__()
    cls='efficientnet-b'+str(compound_coef)
    self.compound_coef = compound_coef
    self._block_args,self._global_params=get_model_params(cls,None)
    self.backbone_net=EfficientNet(self._block_args,self._global_params,extract_features=True)
    self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
    self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
    self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
    self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
    self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
    self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
    conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

    num_anchors = len(self.aspect_ratios) * self.num_scales
    self.bifpn=[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False)
                for _ in range(self.fpn_cell_repeats[compound_coef])]
    self.num_classes = num_classes
    self.backbone_net = EfficientNet(self._block_args,self._global_params,extract_features=True)
    self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
    self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])
  def call(self,inputs):
    features,x=self.backbone_net.call(inputs)
    for _ in range(self.fpn_cell_repeats[self.compound_coef]):
      features=self.bifpn[_].call(features)
    regression = self.regressor.call(features)
    classification = self.classifier(features)
    output=tf.keras.layers.concatenate([classification,regression])
    return output
if __name__ == '__main__':
  from loss import FocalLoss
  model=EfficientDetBackbone(2)
  input1=tf.keras.Input(shape=(768,768,3),batch_size=2)
  output1=model.call(input1)
  c,r=output1
  print(c)
  #mm=tf.keras.Model(inputs=[input1],outputs=[output1,output2])
  #mm.summary()
  #archors=Anchors(anchor_scale=4)
  #archors.get_boxes([768,768])
  
  b=np.array([[[0,0,5,5,3],[5,5,8,8,4],[50,50,300,300,5]],[[0,0,90,90,44]]])
  
  #detloss=FocalLoss()                         
  #l=detloss.call(output2,output1,b)
  #print(l)

