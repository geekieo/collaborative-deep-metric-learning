# -*- coding: utf-8 -*-
'''
@Description: Model modual
@Date: 2019-07-10 17:31:26
@Author: Weng Jingyu
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    """Create the forward propagation section of the graph."""
    raise NotImplementedError()

def prelu(_x, name=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=name, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

class VeNet(BaseModel):
  """Visual Embedding Network
  
  The model is with visual input and embedded output."""

  def create_model(self, model_input, output_size=256, l2_penalty=1e-8,**unused_params):
    """Creates a embedding network with visual feature input.
    Return the inference of the model_input.
    Args:
      model_input: matrix of input features.dimension: [batch, channel, feature],the
        features should be float.
      output_size: size of output embedding. 
    Returns:
      A dictionary with a tensor containing the output of the
      model in the 'output' key. The dimensions of the tensor are
      [batch_size, output_size]."""
    with tf.name_scope("VeNet"):
      # model_input = tf.cast(model_input, tf.float32)
      model_input = tf.nn.l2_normalize(model_input, axis=-1,name='model_input')
      layer_1 = slim.fully_connected(
          model_input, 5000, activation_fn=tf.nn.leaky_relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      layer_2 = slim.fully_connected(
          layer_1, output_size, activation_fn=tf.nn.leaky_relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      l2_norm = tf.nn.l2_normalize(layer_2, axis=-1,name='model_output')
      return {"layer_1":layer_1, "layer_2":layer_2,"l2_norm": l2_norm}

class VedeNet():
  """Visual Embedding and Doc Embedding Network"""
  def create_model(self, model_input, output_size=256, l2_penalty=1e-8):
    """
    Args:
      model_input: matrix of input features.dimension: [batch, channel, feature].
          a feature is concatenate visual feature with doc feature ,and it's float.
      output_size: size of output embedding. """
    with tf.name_scope("VedeNet"):
      # visual module
      visual_input = model_input[:,:1500]   # visual feature vector
      visual_input = tf.nn.l2_normalize(visual_input, axis=-1,name='visual_input')
      layer_visual_1 = slim.fully_connected(
          visual_input, 5000, activation_fn=tf.nn.leaky_relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      layer_visual_2 = slim.fully_connected(
          layer_visual_1, 256, activation_fn=tf.nn.leaky_relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      # doc module
      doc_input = model_input[:,1500:]   # doc feature vector
      doc_input = tf.nn.l2_normalize(doc_input, axis=-1,name='doc_input') 
      layer_doc_1 = slim.fully_connected(
          doc_input, 400, activation_fn=tf.nn.leaky_relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      layer_doc_2 = slim.fully_connected(
          layer_doc_1, 256, activation_fn=tf.nn.leaky_relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      # fusion by multiply
      layer_funsion = tf.multiply(layer_visual_2, layer_doc_2, name="multiply_funsion")
      l2_norm = tf.nn.l2_normalize(layer_funsion, axis=-1,name='model_output')
      return {"l2_norm": l2_norm}
