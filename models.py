# -*- coding: utf-8 -*-
'''
@Description: Model modual
@Date: 2019-07-10 17:31:26
@Author: Weng Jingyu
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim


def prelu(_x, name=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=name, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def fully_connected(input_tensor,
                    output_size,
                    activation_fn=tf.nn.leaky_relu,
                    l2_penalty=1e-8,
                    bias_init=0.1,
                    name=None):
  # bias_init一般为0，但这里不能为0，否则融合模型无法学习
  return slim.fully_connected(input_tensor, output_size, 
                              activation_fn=activation_fn,
                              weights_regularizer=slim.l2_regularizer(l2_penalty),
                              biases_initializer=tf.constant_initializer(bias_init),
                              scope=name)


class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    """Create the forward propagation section of the graph."""
    raise NotImplementedError()


class VNet(BaseModel):
  """Visual Feature Network

  The model is with visual vector input and vector output."""

  def create_model(self, model_input, output_size=256):
    """Creates a neural network with visual feature input.
    Return the inference of the model_input.
    Args:
      model_input: matrix of input features.float.dimension:[batch_size*channel_num, feature_size]
      output_size: size of output vector. 
    Returns:
      A dictionary with a tensor containing the output of the
      model in the 'output' key. The dimensions of the tensor are
      [batch_size, output_size]."""
    with tf.name_scope(self.__class__.__name__):
      # model_input = tf.cast(model_input, tf.float32)
      model_input = tf.nn.l2_normalize(model_input, axis=-1,name='model_input')
      layer_1 = fully_connected(model_input,5000,bias_init=0)
      layer_2 = fully_connected(layer_1, output_size,bias_init=0)
      l2_norm = tf.nn.l2_normalize(layer_2, axis=-1,name='model_output')
      return {"layer_1":layer_1, "layer_2":layer_2,"l2_norm": l2_norm}


class MultiplyNet():
  """A Simple Feautre Fusion Network
  Fusion by multiply
  近邻结果相关度一般
  """

  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output vector. """
    with tf.name_scope(self.__class__.__name__):
      # visual block
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1, name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, bias_init=0.1, name="layer_visual_2")
      # doc block
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input') 
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # fusion by multiply
      layer_fusion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_fusion")
      l2_norm = tf.nn.l2_normalize(layer_fusion, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}

class MlpNet():
  """A Failed  Feautre Fusion Network
  Fusion by multiply, and go through a DNN
  近邻结果相关度较差
  """

  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output vector. """
    with tf.name_scope(self.__class__.__name__):
      # visual block
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, bias_init=0.1, name="layer_visual_2")
      # doc block
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # multiply fusion
      layer_fusion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_fusion")
      # MLP
      layer_fusion_1 = fully_connected(layer_fusion, 600, bias_init=0.1, name="layer_fusion_1")
      layer_fusion_2 = fully_connected(layer_fusion_1, 256, bias_init=0.1, name="layer_fusion_2")
      l2_norm = tf.nn.l2_normalize(layer_fusion_2, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}


class ResNet():
  """A Well Work Feautre Fusion Network. 
  高层网络相同节点的各层之间使用残差连接
  近邻结果相关度很好
  """

  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output vector. """
    with tf.name_scope(self.__class__.__name__):
      # visual block
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, bias_init=0.1, name="layer_visual_2")
      # doc block
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # multiply fusion
      layer_fusion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_fusion")
      # residual MLP
      layer_res_1 = layer_fusion + layer_visual_2 + layer_doc_2
      layer_fusion_1 = fully_connected(layer_res_1, 256, bias_init=0.1, name="layer_fusion_1")
      layer_res_2 = layer_res_1 + layer_fusion_1
      layer_fusion_2 = fully_connected(layer_res_2, 256, bias_init=0.1, name="layer_fusion_2")
      layer_res_3 = layer_res_2 + layer_fusion_2
      l2_norm = tf.nn.l2_normalize(layer_res_3, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}


class DenseNet():
  """A Debugging Feautre Fusion Network.
  高层网络相同节点的各层之间使用稠密连接
  """

  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output vector. """
    with tf.name_scope(self.__class__.__name__):
      model_input = tf.expand_dims(model_input,1) #增加通道维度，用于融合时通道叠加
      # visual block
      visual_input = model_input[:,:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, bias_init=0.1, name="layer_visual_2")
      # doc block
      doc_input = model_input[:,:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # multiply fusion
      layer_fusion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_fusion")
      # after fusion
      layer_dense_0_c = tf.concat([layer_visual_2, layer_doc_2, layer_fusion],1,name="layer_dense_concat")
      print(layer_dense_0_c)
      layer_dense_1 = fully_connected(layer_dense_0_c, 256, name="layer_dense_1") #shape should be: BCW
      layer_dense_1_c = tf.concat([layer_dense_0_c, layer_dense_1],1,name="layer_dense_concat")
      print(layer_dense_1_c)
      layer_dense_2 = fully_connected(layer_dense_1_c, 256, name="layer_dense_2")
      layer_dense_2_c = tf.concat([layer_dense_0_c,layer_dense_1_c, layer_dense_1],1,name="layer_dense_concat")
      print(layer_dense_2_c)
      # transition
      layer_dense_2_c = tf.transpose(layer_dense_2_c,[0,2,1])   # BCW->BWC
      layer_dense_2_c = tf.expand_dims(layer_dense_2_c,1)       # BWC->BHWC: [batch_size,1,256,12]
      channel_num = tf.shape(layer_dense_2_c)
      kernel  = tf.get_variable("reduice_channel_kernel", [1, 1, channel_num, 1],
                initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
      layer_trans = tf.nn.conv2d(layer_dense_2_c, kernel,[1, 1, 1, 1],'SAME',name="layer_transition") # input:BHWC. kernel: HWiCoC
      layer_trans = tf.squeeze(layer_trans,[1,3])
      l2_norm = tf.nn.l2_normalize(layer_trans, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}
      
class ResNetV2():
  """A Debugging Wider ResNet. 
  与V1对比，底层增加2路浅层网络，4路输出交叉融合，
  高层网络相同节点的各层之间使用残差连接
  """
  
  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension: [batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output vector. """
    with tf.name_scope(self.__class__.__name__):
      # visual block
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1_1 = fully_connected(visual_input, 5000, bias_init=.1, name="layer_visual_1_1")
      layer_visual_1_2 = fully_connected(layer_visual_1_1, 256, bias_init=.1, name="layer_visual_1_2")
      layer_visual_2_1 = fully_connected(visual_input, 256, bias_init=.1, name="layer_visual_2_1")
      # doc block
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1_1 = fully_connected(doc_input, 400,  bias_init=.1, name="layer_doc_1_1")
      layer_doc_1_2 = fully_connected(layer_doc_1_1, 256, bias_init=.1, name="layer_doc_1_2")
      layer_doc_2_1 = fully_connected(doc_input, 256, bias_init=.1, name="layer_doc_2_1")
      # multiply fusion
      layer_fusion_1 = tf.multiply(layer_visual_1_2, layer_doc_1_2,"multiply_fusion_1")
      layer_fusion_2 = tf.multiply(layer_visual_1_2, layer_doc_2_1,"multiply_fusion_2")
      layer_fusion_3 = tf.multiply(layer_visual_2_1, layer_doc_1_2,"multiply_fusion_3")
      layer_fusion_4 = tf.multiply(layer_visual_2_1, layer_doc_2_1,"multiply_fusion_4")
      # residual block
      layer_res_1 = layer_fusion_1 + layer_fusion_2 + layer_fusion_3 + layer_fusion_4 +\
                    layer_visual_1_2 + layer_visual_2_1 + layer_doc_1_2 + layer_doc_2_1
      layer_fusion_1 = fully_connected(layer_res_1, 256, name="layer_fusion_1")
      layer_res_2 = layer_res_1 + layer_fusion_1
      layer_fusion_2 = fully_connected(layer_res_2, 256, name="layer_fusion_2")
      layer_res_3 = layer_res_2 + layer_fusion_2
      l2_norm = tf.nn.l2_normalize(layer_res_3, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}

