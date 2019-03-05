# -*- coding:utf-8 -*-
"""Provides input pipe, which can get input data tensors for models."""
import tensorflow as tf

class BasePipe(object):
  """Inherit from this class when implementing new readers."""

  def create_pipe(self, unused_data, **unused_params):
    """Create the section of the graph which reads the training/val/testing data."""
    raise NotImplementedError()

class TripletPipe(BasePipe):

  def create_pipe(self,
                  data,
                  batch_size=1000,
                  num_epochs=None,
                  num_readers=1,
                  **unused_params)
  """Construct a memory data pipe.
    data: ndarray of training sample
  Args:
    num_classes: a positive integer for the number of classes.
    feature_sizes: positive integer(s) for the feature dimensions as a list.
    feature_names: the feature name(s) in the tensorflow record as a list.
  """
  dataset = tf.data.Dataset.from_tensor_slices(data)