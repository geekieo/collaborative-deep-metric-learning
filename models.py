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
    """Creates a embedding network with visual feature input

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      output_size: size of output embedding. 

    Returns:
      A dictionary with a tensor containing the embeddings of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x number of nodes in output layer."""
    layer_1 = slim.fully_connected(
        model_input, 2560, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    layer_2 = slim.fully_connected(
        layer_1, output_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = tf.nn.l2_normalize(layer_2)
    return {"predictions": output}
