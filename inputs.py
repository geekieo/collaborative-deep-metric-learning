# -*- coding:utf-8 -*-
"""Provides input pipe, which can get input data tensors for models."""
import tensorflow as tf
from tensorflow import logging

logging.set_verbosity(logging.INFO)

class BasePipe(object):
  """Inherit from this class when implementing new readers."""

  def create_pipe(self, unused_data, **unused_params):
    """Create the section of the graph which reads the training/val/testing data."""
    raise NotImplementedError()

class TripletPipe(BasePipe):
  
  def create_pipe(self,
                  data=None,
                  filename=None,
                  batch_size=10,
                  num_epochs=None,
                  num_readers=1,
                  buffer_size=1000,
                  **unused_params):
    """Construct a memory data pipe.
    Args:
      data: ndarray of triplet([anchor feature, positive feature, negative feature])
            all training data.
      filename: str. A path pattern to the data files. If 'data' and 'filename'
                both are provided, then only data is used. 'data' and 'filename'
                must have at least one.
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      TODO num_readers: How many I/O threads to use.
    return:
      An Iterator over the elements of this dataset. use Iterator.get_next() to
      get the next batches of features tensor(anchor, positive, negative)
    rasie:
    """
    if data:
      dataset = tf.data.Dataset.from_tensor_slices(data)
      # Transformation: batch,shuffle,repeat
      dataset = dataset.repeat(num_epochs)
      dataset = dataset.batch(batch_size)
      dataset = dataset.shuffle(buffer_size)
      # iterator
      iterator = dataset.make_one_shot_iterator()
      return iterator
    elif filename:
      pass
    else:
      raise ValueError('Input ERROR. Check argument "data" or "filename".')

