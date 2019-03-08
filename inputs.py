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
                  batch_size=10,
                  num_epochs=None,
                  num_readers=1,
                  buffer_size=1000,
                  **unused_params):
    """Construct a memory data pipe.
    Args:
      data: ndarray of triplet([anchor feature, positive feature, negative feature])
        all training data.
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      TODO num_readers: How many I/O threads to use.
    return:
      A tensor containing a batch of features tensor(anchor, positive, negative)
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # Transformation: batch,shuffle,repeat
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size)
    # iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
