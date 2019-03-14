# -*- coding:utf-8 -*-
import tensorflow as tf

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_triplets, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

    Args:
    unused_triplets: a 3-d tensor storing the embeddings. The dimensions are
      [batch, triplet_channel, embedding]. The default labels of each
      triplet are [anchor, positive, negative].

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()

class HingeLoss(BaseLoss):
  def calculate_loss(self, triplets, margin=1):
    """ 
    Args:
      triplets: 3-d tensor storing the triplets. The dimensions are
        [batch, triplet_channel, embedding]. The default triplet_channel is 3.
        The default labels of each triplet are [anchor embedding, positive 
        embedding, negative embedding].
      margin: int. The margin for triplet loss
    Retrun:
      A scalar loss tensor.
    """
    with tf.name_scope("loss_hinge"):
      triplets = tf.cast(triplets, tf.float32)
      anchors, positives, negatives = tf.split(triplets, 3, axis=1)
      pos_dist = tf.square(tf.subtract(anchors,positives), name="pos_dist")
      neg_dist = tf.square(tf.subtract(anchors,negatives), name="neg_dist")
      hinge_dist = tf.maximum(pos_dist - neg_dist + margin, 0.0, name="hinge_dist")
      hinge_loss = tf.reduce_mean(hinge_dist, name="hinge_loss")
      return hinge_loss