import tensorflow as tf
import itertools
import numpy as np
from tensorflow.keras import backend as k
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
import os
block_args,global_params=get_model_params('efficientnet-b2',None)
os.environ['CUDA_VISIBLE_DEVICES'] = " "
class MBConvBlock(tf.keras.layers.Layer):
  def __init__(self, block_args, global_params, name=None):
      super().__init__(name=name)
      self._block_args = block_args
      self._bn_mom = global_params.batch_norm_momentum
      self._bn_eps = global_params.batch_norm_epsilon
      #print(self._bn_mom,self._bn_eps)
      self.has_se = None #(self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
      self.id_skip = block_args.id_skip  # skip connection and drop connect
      self._data_format = global_params.data_format
      self._channel_axis = 1 if self._data_format == 'channels_first' else -1
      
      self._relu_fn = tf.keras.layers.ELU()
      self.vars=[]    #test-var
      self._build()
      
  def get_config(self):
      config ={"_block_args":self._block_args,"_bn_mom":self._bn_mom,"_bn_eps":self._bn_eps,"has_se":self.has_se,"id_skip":self.id_skip,"_data_format": self._data_format,"_channel_axis":self._channel_axis}
      base_config = super(MBConvBlock, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))
      # Expansion phase
  def _build(self):
      inp = self._block_args.input_filters  # number of input channels
      oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
      """Builds block according to the arguments."""
      # pylint: disable=g-long-lambda
      bid = itertools.count(0)
      get_bn_name = lambda: 'tpu_batch_normalization' + ('' if not next(
           bid) else '_' + str(next(bid) // 2))
      cid = itertools.count(0)
      get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
           next(cid) // 2))
    # pylint: enable=g-long-lambda
      #self.vars.append(self.add_weight(initializer='random_normal')) #testvar
      if self._block_args.expand_ratio != 1:
         self._expand_conv = tf.keras.layers.Conv2D(
               filters=oup,
               kernel_size=[1, 1],
               strides=[1, 1],
               kernel_initializer='normal',
               padding='same',
               data_format=self._data_format,
               use_bias=False,
               name=get_conv_name())
         self._bn0=tf.keras.layers.BatchNormalization(axis=self._channel_axis,
               momentum=self._bn_mom,
               epsilon=self._bn_eps,
               name=get_bn_name())
      # Depthwise convolution phase
      k = self._block_args.kernel_size
      s = self._block_args.stride  
      if isinstance(s, list):
         s = s[0]
      self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
             kernel_size=[k,k],
             strides=s,
             depthwise_initializer='normal',
             padding='same',
             data_format=self._data_format,
             use_bias=False,
             name='depthwise_conv2d')
      self._bn1 =tf.keras.layers.BatchNormalization(axis=self._channel_axis,
               momentum=self._bn_mom,
               epsilon=self._bn_eps,
               name=get_bn_name())
      if self.has_se:
         self._local_pooling=tf.keras.layers.AveragePooling2D
         num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
         self._se_reduce = tf.keras.layers.Conv2D(
               num_squeezed_channels,
               kernel_size=[1, 1],
               strides=[1, 1],
               kernel_initializer='normal',
               padding='same',
               data_format=self._data_format,
               use_bias=True,
               name='conv2d')
         self._se_expand =self._se_expand = tf.keras.layers.Conv2D(
                oup,
       		kernel_size=[1, 1],
        	strides=[1, 1],
        	kernel_initializer='normal',
        	padding='same',
        	data_format=self._data_format,
        	use_bias=True,
        	name='conv2d_1')
      final_oup = self._block_args.output_filters
      self._project_conv = tf.keras.layers.Conv2D(
        	filters=final_oup,
        	kernel_size=[1, 1],
        	strides=[1, 1],
        	kernel_initializer='normal',
        	padding='same',
        	data_format=self._data_format,
        	use_bias=False,
        	name=get_conv_name())
      self._bn2 =tf.keras.layers.BatchNormalization(axis=self._channel_axis,
               momentum=self._bn_mom,
               epsilon=self._bn_eps,
               name=get_bn_name())

  def call(self, inputs):
      x = inputs
      
      if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._relu_fn(x)
            

      x = self._depthwise_conv(x)
      x = self._bn1(x)
      x = self._relu_fn(x)
      
      # Squeeze and Excitation
      if self.has_se:
          x_squeezed = self._local_pooling(pool_size=(x.shape[1],x.shape[2]),padding='valid')(x)
          x_squeezed = self._se_reduce(x_squeezed)
          x_squeezed = self._relu_fn(x_squeezed)
          x_squeezed = self._se_expand(x_squeezed)
          
          x = tf.keras.activations.sigmoid(x)

      x = self._project_conv(x)
      x = self._bn2(x)
      
      #x = tf.math.multiply(self.vars[-1],x)  #test_var
      # Skip connection and drop connect
      input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
      if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
        
          x =tf.keras.layers.add([x,inputs])  # skip connection
      return x  

    
class EfficientNet(tf.keras.Model):
  def __init__(self, blocks_args=None, global_params=None,extract_features=False):
    super().__init__()
    assert isinstance(blocks_args, list), 'blocks_args should be a list'
    assert len(blocks_args) > 0, 'block args must be greater than 0'
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._extract_features=extract_features
    self._build()
  def _build(self):
    bn_mom =  self._global_params.batch_norm_momentum
    bn_eps = self._global_params.batch_norm_epsilon
    # Stem
    in_channels = 3  # rgb
    out_channels = round_filters(32, self._global_params)  # number of output channels
    self._blocks = []
    self._conv_stem =tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer='normal',
        padding='same',
        data_format=global_params.data_format,
        use_bias=False)
    self._bn0=tf.keras.layers.BatchNormalization(axis=(1 if global_params.data_format == 'channels_first' else -1),
               momentum=bn_mom,
               epsilon=bn_eps,
               )
    # Builds blocks.
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for block_args in self._blocks_args:
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
           input_filters=round_filters(block_args.input_filters, self._global_params),
           output_filters=round_filters(block_args.output_filters, self._global_params),
           num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
        self._blocks.append(MBConvBlock(block_args, self._global_params))
        if block_args.num_repeat > 1:
           block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
        for _ in range(block_args.num_repeat - 1):
            self._blocks.append(MBConvBlock(block_args, self._global_params))
    # Head
    in_channels = block_args.output_filters  # output of final block
    out_channels = round_filters(1280, self._global_params)
    self._conv_head=tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format=global_params.data_format,
        use_bias=False)
    self._bn1 =tf.keras.layers.BatchNormalization(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=bn_mom,
        epsilon=bn_eps,
               )
    # Final linear layer
    if  self._extract_features:
      self._conv_extract=tf.keras.layers.Conv2D(
          filters=32,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format=global_params.data_format,
          use_bias=False)
      self._bn2 =tf.keras.layers.BatchNormalization(
        axis=(1 if global_params.data_format == 'channels_first' else -1),
        momentum=bn_mom,
        epsilon=bn_eps,
                )
    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=global_params.data_format)
    self._dropout = tf.keras.layers.Dropout(global_params.dropout_rate)
    self._fc = tf.keras.layers.Dense(
          101,
          kernel_initializer='normal')     #1111111111111111111111111111111111111111111111111111111111111111111111
    self._fc2=  tf.keras.layers.Dense(
          51,
          kernel_initializer='normal')
    
    self._relu_fn = tf.keras.layers.ELU()
    self._softmax=tf.keras.layers.Softmax()
    self._upsample=tf.image.resize
    
    self.se=[tf.keras.layers.Conv2D(filters=50,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
    
  def call(self,inputs):
    # Stem
    
    inputs=tf.keras.layers.Lambda(lambda x : x/255.,input_shape=(inputs.shape[1],inputs.shape[2],inputs.shape[3]))(inputs)
    x=self._relu_fn(self._bn0(self._conv_stem(inputs)))
    feature_maps = []
    # Blocks
    for idx, block in enumerate(self._blocks):
            
         
        x = block(x)
        
        if block._depthwise_conv.strides == (2, 2):
           feature_maps.append(last_x)
                
        elif idx == len(self._blocks) - 1:
           feature_maps.append(x)
        last_x = x
    # Head
    x = self._relu_fn(self._bn1(self._conv_head(x)))
   
    # Pooling and final linear layer
    if not self._extract_features:
     # x = self._avg_pooling(x)
     return x  
     # x = self._dropout(x)
     # y = self._fc2(x)
     # x = self._fc(x)
      #x = self._softmax(x)
    else:
      x = self._relu_fn(self._bn2(self._conv_extract(x)))
      #print(x.shape)
      
      
      #x = self._upsample(x,[x.shape[1]*2,x.shape[2]*2],method='nearest')
      x = tf.keras.layers.Flatten()(x)
   
      return x
  
class PAM_Module(tf.keras.layers.Layer):
  def __init__(self, in_dim):
        super(PAM_Module,self).__init__()
        self.chanel_in = in_dim
		# 分别得到B,C,D，这里对B和C的输出通道数进行了压缩8倍
        self.query_conv = tf.keras.layers.Conv2D(
          filters=in_dim//8.,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='valid',
          data_format='channels_last',
          use_bias=True)
        self.key_conv = tf.keras.layers.Conv2D(
          filters=in_dim//8.,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='valid',
          data_format='channels_last',
          use_bias=True)
        self.value_conv = tf.keras.layers.Conv2D(
          filters=in_dim,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='valid',
          data_format='channels_last',
          use_bias=True)
        # gamma 对应上述的alpha
        self.gamma = tf.Variable(0.,trainable=True)

        self.softmax = tf.keras.layers.Softmax()
  def call(self, x):
        """
            inputs :
                x : input feature maps(B X H X W X C)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
       
        m_batchsize= tf.shape(x)[0]
        height=tf.shape(x)[1]
        width=tf.shape(x)[2]
        C=tf.shape(x)[3]
        #x=tf.transpose(x,perm=[0,3,1,2])
        # 矩阵B
       
        proj_query = self.query_conv(x) #B x H x W x C/8

        proj_query = tf.transpose(proj_query,perm=[0,3,1,2]) # B x C/8 x H x W
    
        proj_query = tf.reshape(proj_query,[m_batchsize,-1,width*height]) # B x C/8 x (HxW)
   
        proj_query=tf.transpose(proj_query,perm=[0,2,1]) # B x (HxW) x C/8
        
        # 矩阵C
        proj_key = tf.reshape(tf.transpose(self.key_conv(x),perm=[0,3,1,2]),[m_batchsize,-1,width*height]) #B x C/8 x (HxW)
        
        # torch.bmm点积操作
        energy = tf.matmul(proj_query, proj_key) # B x (HxW) x (HxW)
        # 映射到0-1区间的系数
        attention = self.softmax(energy)
        proj_value = tf.reshape(tf.transpose(self.value_conv(x),perm=[0,3,1,2]),[m_batchsize,-1,width*height]) # B x C x (w*h) .B x (w*h) x (w*h)
        out = tf.matmul(proj_value, tf.transpose(attention,perm=[0,2,1]))
        
        #print(out,x.shape)
        # 矩阵D
        out = tf.reshape(out,[m_batchsize,C, height, width])
        out = tf.transpose(out,perm=[0,2,3,1])
        out = self.gamma*out + x
        #out = tf.transpose(out,perm=[0,2,3,1])
        return out


class CAM_Module(tf.keras.layers.Layer):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module,self).__init__()
        


        self.gamma = tf.Variable(0.,trainable=True)
        self.softmax = tf.keras.layers.Softmax()
    def call(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize= tf.shape(x)[0]
        height=tf.shape(x)[1]
        width=tf.shape(x)[2]
        C=tf.shape(x)[3]
        #print(m_batchsize,height,width,C)
        x=tf.transpose(x,perm=[0,3,1,2])
        proj_query = tf.reshape(x,[m_batchsize, C, -1])
        proj_key = tf.reshape(x,[m_batchsize, C, -1])
        proj_key = tf.transpose(proj_key,perm=[0,2,1])
        energy = tf.matmul(proj_query, proj_key)
        #print(energy)
        energy_new = tf.reduce_max(energy,axis=-1, keepdims=True)
        #print(energy_new)
        energy_new=energy_new-energy
        attention = self.softmax(energy_new)
        proj_value = tf.reshape(x,[m_batchsize, C, -1])

        out =tf.matmul(attention, proj_value)
        out = tf.reshape(out,[m_batchsize, C, height, width])

        out = self.gamma*out + x
        out = tf.transpose(out,perm=[0,2,3,1])
        return out




class DANet(tf.keras.Model):
  def __init__(self, in_channels, out_channels):
    super(DANet, self).__init__()
    #print(in_channels,out_channels)
    inter_channels = in_channels // 4
    self.conv5a =tf.keras.layers.Conv2D(
          filters=inter_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
    self.bn1=tf.keras.layers.BatchNormalization(
        axis=-1)
    self._relu_fn = tf.keras.layers.ReLU()
    self.conv5c=tf.keras.layers.Conv2D(
          filters=inter_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
    self.bn2=tf.keras.layers.BatchNormalization(
        axis=-1)
    self.sa = PAM_Module(inter_channels)
    self.sc = CAM_Module()
    self.conv51=tf.keras.layers.Conv2D(
          filters=inter_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
    self.conv52=tf.keras.layers.Conv2D(
          filters=inter_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
    self.bn3=tf.keras.layers.BatchNormalization(
        axis=-1)
    self.bn4=tf.keras.layers.BatchNormalization(
        axis=-1)
    self.dropout=tf.keras.layers.SpatialDropout2D(0.1, data_format='channels_last')
    self.conv6=tf.keras.layers.Conv2D(
          filters=out_channels,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
    self.conv7=tf.keras.layers.Conv2D(
          filters=out_channels,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
    self.conv8=tf.keras.layers.Conv2D(
          filters=out_channels,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer='normal',
          padding='same',
          data_format='channels_last',
          use_bias=False)
  def call(self, x):
      feat1 =  self._relu_fn(self.bn1(self.conv5a(x)))
       
      sa_feat = self.sa.call(feat1)
      sa_conv = self._relu_fn(self.bn3(self.conv51(sa_feat)))
      sa_output = self.conv6(self.dropout(sa_conv))

      feat2 = self._relu_fn(self.bn2(self.conv5c(x)))
      sc_feat = self.sc(feat2)
      sc_conv = self._relu_fn(self.bn4(self.conv52(sc_feat)))
      sc_output = self.conv7(self.dropout(sc_conv))
      
      feat_sum = sa_conv+sc_conv
        
      sasc_output = self.conv8(self.dropout(feat_sum))

        
      return sasc_output
class Ypdanet(tf.keras.Model):
  def __init__(self,compound_coef,train=True):
    super().__init__()
    cls='efficientnet-b'+str(compound_coef)
    self._block_args,self._global_params=get_model_params(cls,None)
    self.backbone_net=EfficientNet(self._block_args,self._global_params)
    self.train=train
    self._build()
  def _build(self):
    self._relu=tf.keras.activations.relu
    self.dropout0=tf.keras.layers.Dropout(0.2)
    self.dropout1=tf.keras.layers.Dropout(0.5)
    self.path_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense')
    self.path_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense2')
    self.path_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense3')
    self.path_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_dense4')
    self.path_dense5=tf.keras.layers.Dense(51,kernel_initializer='normal',name='path_dense51') # 50 points 1 vaild_len
    self.weight_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight')
    self.weight_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight1')
    self.weight_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight2')
    self.weight_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_weight3')
    self.weight_dense5=tf.keras.layers.Dense(50,kernel_initializer='normal',name='path_weight4') #dRel,yRel,vRel,aRel.prob
    self.lead_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense')
    self.lead_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense2')
    self.lead_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense3')
    self.lead_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='lead_dense4')
    self.lead_dense5=tf.keras.layers.Dense(4,kernel_initializer='normal',name='lead_dense51') # 
    self.DA=DANet(512,32)
  def call(self,input1):
    x=self.backbone_net.call(input1)
    #x_a=tf.keras.layers.Flatten()(x)
    x=self.DA.call(x)
    output = tf.keras.layers.Flatten()(x)
    output=self._relu(output)
     #path
    output1=self.path_dense(output)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense2(output1)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense3(output1)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense4(output1)
    output1=self._relu(output1)
    output1=self.dropout1(output1)
    output1=self.path_dense5(output1)
#lead
    output2=self.lead_dense(output)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense2(output2)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense3(output2)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense4(output2)
    output2=self._relu(output2)
    output2=self.dropout1(output2)
    output2=self.lead_dense5(output2)
    #weight
    output3=self.weight_dense(output)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense2(output3)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense3(output3)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense4(output3)
    output3=self._relu(output3)
    output3=self.dropout1(output3)
    output3=self.weight_dense5(output3)
	
    o=tf.keras.layers.concatenate([output1,output3,output2],axis=-1)
        
    return o
    
class Ypdanet2(tf.keras.Model):
  def __init__(self,compound_coef,train=True):
    super().__init__()
    cls='efficientnet-b'+str(compound_coef)
    self._block_args,self._global_params=get_model_params(cls,None)
    self.backbone_net=EfficientNet(self._block_args,self._global_params)
    self.train=train
    self._build()
  def _build(self):
    self.pre_gru_dense=tf.keras.layers.Dense(1024,kernel_initializer='normal',name='pre_gru_dense')
   
    self.rnn_r=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_r')
    self.rnn_z=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_z')
    self.rnn_h=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_h')
    self.rnn_rr=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_rr')
    self.rnn_rz=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_rz')
    self.snpe_pleaser=tf.keras.layers.Dense(512,kernel_initializer='normal',name='snpe_pleaser')
    self.rnn_rh=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_rh')

    self.one_minus=tf.keras.layers.Dense(512,kernel_initializer='normal',name='one_minus')
    self.add=tf.keras.layers.add
    self.sigmoid=tf.keras.activations.sigmoid
    self.multiply=tf.keras.layers.multiply
    self.tanh=tf.keras.activations.tanh
    self._relu=tf.keras.activations.relu
    self.dropout0=tf.keras.layers.Dropout(0.2)
    self.dropout1=tf.keras.layers.Dropout(0.5)
    self.path_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense')
    self.path_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense2')
    self.path_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense3')
    self.path_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_dense4')
    self.path_dense5=tf.keras.layers.Dense(51,kernel_initializer='normal',name='path_dense51') # 50 points 1 vaild_len
    self.weight_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight')
    self.weight_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight1')
    self.weight_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight2')
    self.weight_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_weight3')
    self.weight_dense5=tf.keras.layers.Dense(50,kernel_initializer='normal',name='path_weight4')
    self.lead_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense')
    self.lead_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense2')
    self.lead_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense3')
    self.lead_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='lead_dense4')
    self.lead_dense5=tf.keras.layers.Dense(4,kernel_initializer='normal',name='lead_dense51') # 
    self.DA=DANet(512,32)
  def call(self,input1,input2):
    x=self.backbone_net.call(input1)
 
    x=self.DA.call(x)
    x = tf.keras.layers.Flatten()(x)
    h=input2
        
    x=self.pre_gru_dense(x)
        
    x=self._relu(x)
    
    rnn_r=self.rnn_r(x)
    rnn_z=self.rnn_z(x)
    rnn_h=self.rnn_h(x)
    rnn_rr=self.rnn_rr(h)
    rnn_rz=self.rnn_rz(h)
    snpe_pleaser=self.snpe_pleaser(h)
    rnn_rh=self.rnn_rh(snpe_pleaser)
    
    r=self.add([rnn_r,rnn_rr])
    z=self.add([rnn_z,rnn_rz])
    r=self.sigmoid(r)
    rnn_rh=self.multiply([rnn_rh,r])
    rnn_rh=self.add([rnn_rh,rnn_h])
    rnn_rh=self.tanh(rnn_rh)
    z=self.sigmoid(z)
    ht=self.multiply([snpe_pleaser,z])
    h=self.multiply([rnn_rh,self.one_minus(z)])
    #state
    output=self.add([h,ht])
     #path
    output1=self.path_dense(output)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense2(output1)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense3(output1)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense4(output1)
    output1=self._relu(output1)
    output1=self.dropout1(output1)
    output1=self.path_dense5(output1)
#lead
    output2=self.lead_dense(output)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense2(output2)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense3(output2)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense4(output2)
    output2=self._relu(output2)
    output2=self.dropout1(output2)
    output2=self.lead_dense5(output2)
    #weight
    output3=self.weight_dense(output)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense2(output3)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense3(output3)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense4(output3)
    output3=self._relu(output3)
    output3=self.dropout1(output3)
    output3=self.weight_dense5(output3)
	#path+lead+state 51+8+512
    o=tf.keras.layers.concatenate([output1,output3,output2,output],axis=-1)
        
    return o

class Ypgru(tf.keras.Model):
  def __init__(self,compound_coef,train=True):
    super().__init__()
    cls='efficientnet-b'+str(compound_coef)
    self._block_args,self._global_params=get_model_params(cls,None)
    self.backbone_net=EfficientNet(self._block_args,self._global_params,extract_features=True)
    self.train=train
    self._build()
  def _build(self):
    self.pre_gru_dense=tf.keras.layers.Dense(1024,kernel_initializer='normal',name='pre_gru_dense')
    self._relu=tf.keras.activations.relu
    self.rnn_r=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_r')
    self.rnn_z=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_z')
    self.rnn_h=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_h')
    self.rnn_rr=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_rr')
    self.rnn_rz=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_rz')
    self.snpe_pleaser=tf.keras.layers.Dense(512,kernel_initializer='normal',name='snpe_pleaser')
    self.rnn_rh=tf.keras.layers.Dense(512,kernel_initializer='normal',name='rnn_rh')
    self.add=tf.keras.layers.add
    self.sigmoid=tf.keras.activations.sigmoid
    self.multiply=tf.keras.layers.multiply
    self.tanh=tf.keras.activations.tanh
    self.path_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense')
    self.path_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense2')
    self.path_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense3')
    self.path_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_dense4')
    self.path_dense5=tf.keras.layers.Dense(101,kernel_initializer='normal',name='path_dense5') # 50 points 1 vaild_len
    self.lead_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight')
    self.lead_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight2')
    self.lead_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight3')
    self.lead_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_weight4')
    self.lead_dense5=tf.keras.layers.Dense(9,kernel_initializer='normal',name='path_weight5') #dRel,yRel,vRel,aRel.prob
  def call(self,input1,input2):
      
        
        x=self.backbone_net.call(input1)
        
        h=input2
        
        x=self.pre_gru_dense(x)
        
        x=self._relu(x)
    
        rnn_r=self.rnn_r(x)
        rnn_z=self.rnn_z(x)
        rnn_h=self.rnn_h(x)
        rnn_rr=self.rnn_rr(h)
        rnn_rz=self.rnn_rz(h)
        snpe_pleaser=self.snpe_pleaser(h)
        rnn_rh=self.rnn_rh(snpe_pleaser)
    
        r=self.add([rnn_r,rnn_rr])
        z=self.add([rnn_z,rnn_rz])
        r=self.sigmoid(r)
        rnn_rh=self.multiply([rnn_rh,r])
        rnn_rh=self.add([rnn_rh,rnn_h])
        rnn_rh=self.tanh(rnn_rh)
        z=self.sigmoid(z)
        ht=self.multiply([snpe_pleaser,z])
        h=self.multiply([rnn_rh,1-z])	
        #state
        output=self.add([h,ht])
        #path
        output1=self.path_dense(output)
        output1=self._relu(output1)
        output1=self.path_dense2(output1)
        output1=self._relu(output1)
        output1=self.path_dense3(output1)
        output1=self._relu(output1)
        output1=self.path_dense4(output1)
        output1=self._relu(output1)
        output1=self.path_dense5(output1)
        #lead
        output2=self.lead_dense(output)
        output2=self._relu(output2)
        output2=self.lead_dense2(output2)
        output2=self._relu(output2)
        output2=self.lead_dense3(output2)
        output2=self._relu(output2)
        output2=self.lead_dense4(output2)
        output2=self._relu(output2)
        output2=self.lead_dense5(output2)
        #path+lead+state 101+8+512
        output=tf.keras.layers.concatenate([output1,output2,output],axis=-1)
        return output
class Ypreg(tf.keras.Model):
  def __init__(self,compound_coef,train=True):
    super().__init__()
    cls='efficientnet-b'+str(compound_coef)
    self._block_args,self._global_params=get_model_params(cls,None)
    self.backbone_net=EfficientNet(self._block_args,self._global_params,extract_features=True)
    self.train=train
    self._build()
  def _build(self):
    self._relu=tf.keras.activations.relu
    self.dropout0=tf.keras.layers.Dropout(0.2)
    self.dropout1=tf.keras.layers.Dropout(0.5)
    self.path_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense')
    self.path_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense2')
    self.path_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_dense3')
    self.path_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_dense4')
    self.path_dense5=tf.keras.layers.Dense(51,kernel_initializer='normal',name='path_dense51') # 50 points 1 vaild_len
    self.weight_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight')
    self.weight_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight1')
    self.weight_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='path_weight2')
    self.weight_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='path_weight3')
    self.weight_dense5=tf.keras.layers.Dense(50,kernel_initializer='normal',name='path_weight4') #dRel,yRel,vRel,aRel.prob
    self.lead_dense=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense')
    self.lead_dense2=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense2')
    self.lead_dense3=tf.keras.layers.Dense(256,kernel_initializer='normal',name='lead_dense3')
    self.lead_dense4=tf.keras.layers.Dense(128,kernel_initializer='normal',name='lead_dense4')
    self.lead_dense5=tf.keras.layers.Dense(4,kernel_initializer='normal',name='lead_dense51') # 
    #self.DA=DANet(512,32)
  def call(self,input1):
    x=self.backbone_net.call(input1)
 
    #x=self.DA.call(x)
    output = tf.keras.layers.Flatten()(x)
     #path
    output1=self.path_dense(output)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense2(output1)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense3(output1)
    output1=self._relu(output1)
    output1=self.dropout0(output1)
    output1=self.path_dense4(output1)
    output1=self._relu(output1)
    output1=self.dropout1(output1)
    output1=self.path_dense5(output1)
#lead
    output2=self.lead_dense(output)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense2(output2)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense3(output2)
    output2=self._relu(output2)
    output2=self.dropout0(output2)
    output2=self.lead_dense4(output2)
    output2=self._relu(output2)
    output2=self.dropout1(output2)
    output2=self.lead_dense5(output2)
    #weight
    output3=self.weight_dense(output)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense2(output3)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense3(output3)
    output3=self._relu(output3)
    output3=self.dropout0(output3)
    output3=self.weight_dense4(output3)
    output3=self._relu(output3)
    output3=self.dropout1(output3)
    output3=self.weight_dense5(output3)
	#path+lead+state 51+8+512
    o=tf.keras.layers.concatenate([output1,output3,output2],axis=-1)
        
    return o
if __name__ == '__main__':
  block_args,global_params=get_model_params('efficientnet-b2',None)    
  #model=YpNet(0)
  #num_channels=112
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
  #bifpn=[BiFPN(num_channels,conv_channel_coef[2],first_time=True),BiFPN(num_channels,conv_channel_coef[2]),BiFPN(num_channels,conv_channel_coef[2])]
  model=Ypreg(2)
  
  input1=tf.keras.Input(shape=(160,320,3),name='img')
  input2=tf.keras.Input(shape=(512),name='rnn')
  output=model.call(input1)
  model =tf.keras.Model(inputs=[input1],outputs=output)
          
  model.summary()
  
  
  
  
 
  #mm.save('EfficientNet-d0.h5')
  #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])



         
