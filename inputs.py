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
                  triplets=None,
                  batch_size=10,
                  num_epochs=None,
                  num_readers=1,
                  buffer_size=1000):
    """Construct a memory data pipe.
    Args:
      triplets: ndarray of guid_triplets([anchor_guid, pos_guid, neg_guid]).
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      TODO num_readers: How many I/O threads to use.
    return:
      An Iterator over the elements of this dataset. use Iterator.get_next() to
      get the next batches of features tensor(anchor, positive, negative)
    rasie:
    """
    dataset = tf.data.Dataset.from_tensor_slices(triplets)
    # Transformation: batch,shuffle,repeat
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size)
    # iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator


