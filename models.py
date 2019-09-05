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
                              bias_initializer=tf.constant_initializer(bias_init),
                              scope=name)


class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    """Create the forward propagation section of the graph."""
    raise NotImplementedError()


class VeNet(BaseModel):
  """Visual Embedding Network
  
  The model is with visual input and embedded output."""

  def create_model(self, model_input, output_size=256):
    """Creates a embedding network with visual feature input.
    Return the inference of the model_input.
    Args:
      model_input: matrix of input features.float.dimension:[batch_size*channel_num, feature_size]
      output_size: size of output embedding. 
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


class VedeFusionNet():
  """Visual Embedding and Doc Embedding Network
  近邻结果相关度一般
  """
  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output embedding. """
    with tf.name_scope(self.__class__.__name__):
      # visual module
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1, bias_init=0.1, name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, name="layer_visual_2")
      # doc module
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input') 
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # fusion by multiply
      layer_funsion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_funsion")
      l2_norm = tf.nn.l2_normalize(layer_funsion, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}

class VedeMlpNet():
  """Visual Embedding and Doc Embedding Network
  近邻结果相关度较差
  """
  def create_model(self, model_input, output_size=256):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output embedding. """
    with tf.name_scope(self.__class__.__name__):
      # visual module
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, bias_init=0.1, name="layer_visual_2")
      # doc module
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # fusion by multiply
      layer_funsion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_funsion")
      # MLP
      layer_funsion_1 = fully_connected(layer_funsion, 600, bias_init=0.1, name="layer_funsion_2")
      layer_funsion_2 = fully_connected(layer_funsion_1, 256, bias_init=0.1, name="layer_funsion_2")
      l2_norm = tf.nn.l2_normalize(layer_funsion_2, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}


class VedeResNet():
  """Visual Embedding and Doc Embedding Network
  高层网络相同节点的各层之间使用残差连接
  近邻结果相关度很好
  """
  def create_model(self, model_input, output_size=256, l2_penalty=1e-8):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output embedding. """
    with tf.name_scope(self.__class__.__name__):
      # visual module
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1 = fully_connected(visual_input, 5000, bias_init=0.1, name="layer_visual_1")
      layer_visual_2 = fully_connected(layer_visual_1, 256, bias_init=0.1, name="layer_visual_2")
      # doc module
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1 = fully_connected(doc_input, 400, bias_init=0.1, name="layer_doc_1")
      layer_doc_2 = fully_connected(layer_doc_1, 256, bias_init=0.1, name="layer_doc_2")
      # fusion by multiply
      layer_funsion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_funsion")
      # residual MLP
      layer_res_1 = layer_funsion + layer_visual_2 + layer_doc_2
      layer_funsion_1 = fully_connected(layer_res_1, 256, bias_init=0.1, name="layer_funsion_2")
      layer_res_2 = layer_res_1 + layer_funsion_1
      layer_funsion_2 = fully_connected(layer_res_2, 256, bias_init=0.1, name="layer_funsion_2")
      layer_res_3 = layer_res_2 + layer_funsion_2
      l2_norm = tf.nn.l2_normalize(layer_res_3, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}


class VedeResNetV2():
  """Visual Embedding and Doc Embedding Network
  高层网络相同节点的各层之间使用残差连接
  近邻结果相关度很好
  """
  def create_model(self, model_input, output_size=256, l2_penalty=1e-8):
    """
    Args:
      model_input: matrix of input features.dimension: [batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output embedding. """
    with tf.name_scope(self.__class__.__name__):
      # visual module
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1_1 = fully_connected(visual_input, 5000, bias_init=1, name="layer_visual_1_1")
      layer_visual_1_2 = fully_connected(layer_visual_1_1, 256, bias_init=1, name="layer_visual_1_2")
      layer_visual_2_1 = fully_connected(visual_input, 256, bias_init=1, name="layer_visual_2_1")
      # doc module
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1_1 = fully_connected(doc_input, 400,  bias_init=1, name="layer_doc_1_1")
      layer_doc_1_2 = fully_connected(layer_doc_1_1, 256, bias_init=1, name="layer_doc_1_2")
      layer_doc_2_1 = fully_connected(doc_input, 256, bias_init=1, name="layer_doc_2_1")
      # dense funsion
      layer_funsion_m = layer_visual_1_2 * layer_visual_2_1 * layer_doc_1_2 * layer_doc_2_1
      # residual MLP
      layer_res_1 = layer_funsion_m + layer_visual_1_2 + layer_visual_2_1 + layer_doc_1_2 + layer_doc_2_1
      layer_funsion_1 = fully_connected(layer_res_1, 256, name="layer_funsion_2")
      layer_res_2 = layer_res_1 + layer_funsion_1
      layer_funsion_2 = fully_connected(layer_res_2, 256, name="layer_funsion_2")
      layer_res_3 = layer_res_2 + layer_funsion_2
      l2_norm = tf.nn.l2_normalize(layer_res_3, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}


class VedeDenseNet():
  """Visual Embedding and Doc Embedding Network
  高层网络相同节点的各层之间使用稠密连接
  """
  def create_model(self, model_input, output_size=256, l2_penalty=1e-8):
    """
    Args:
      model_input: matrix of input features.dimension:[batch_size*channel_num, feature_size].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output embedding. """
    with tf.name_scope(self.__class__.__name__):
      # visual module
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1_1 = fully_connected(visual_input, 5000, bias_init=.1, name="layer_visual_1_1")
      layer_visual_1_2 = fully_connected(layer_visual_1_1, 256, bias_init=.1, name="layer_visual_1_2")
      layer_visual_2_1 = fully_connected(visual_input, 256, bias_init=.1, name="layer_visual_2_1")
      # doc module
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input')
      layer_doc_1_1 = fully_connected(doc_input, 400,  bias_init=.1, name="layer_doc_1_1")
      layer_doc_1_2 = fully_connected(layer_doc_1_1, 256, bias_init=.1, name="layer_doc_1_2")
      layer_doc_2_1 = fully_connected(doc_input, 256, bias_init=.1, name="layer_doc_2_1")
      # dense connect
      

      layer_funsion_m = layer_visual_1_2 * layer_visual_2_1 * layer_doc_1_2 * layer_doc_2_1
      # residual MLP
      layer_res_1 = layer_funsion_m + layer_visual_1_2 + layer_visual_2_1 + layer_doc_1_2 + layer_doc_2_1
      layer_funsion_1 = fully_connected(layer_res_1, 600, name="layer_funsion_2")
      layer_res_2 = layer_res_1 + layer_funsion_1
      layer_funsion_2 = fully_connected(layer_res_2, 256, name="layer_funsion_2")
      layer_res_3 = layer_res_2 + layer_funsion_2
      l2_norm = tf.nn.l2_normalize(layer_res_3, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}

class VedeNet():
  """Visual Embedding and Doc Embedding Network"""
  def create_model(self, model_input, output_size=256, l2_penalty=1e-8):
    VedeDenseNet(self, model_input, output_size=256, l2_penalty=1e-8)