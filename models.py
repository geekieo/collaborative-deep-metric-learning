# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    """Create the forward propagation section of the graph."""
    raise NotImplementedError()

class VENet(BaseModel):
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
    with tf.name_scope("VENet"):
      # model_input = tf.cast(model_input, tf.float32)
      layer_1 = slim.fully_connected(
          model_input, 4000, activation_fn=tf.nn.leaky_relu,
          weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          biases_initializer = tf.constant_initializer(0.1))
      layer_2 = slim.fully_connected(
          layer_1, output_size, activation_fn=tf.nn.leaky_relu,
          weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          biases_initializer = tf.constant_initializer(0.1))
      l2_norm = tf.nn.l2_normalize(layer_2, axis=-1)
      return {"layer_1":layer_1, "layer_2":layer_2,"l2_norm": l2_norm}

