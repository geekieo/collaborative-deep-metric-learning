# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np

from imitation_data import gen_unique_id_array
from imitation_data import gen_all_watched_guids
from imitation_data import gen_features
from parse_data import get_all_cowatch
from parse_data import get_triplet
from losses import HingeLoss

class test_losses():

  def __init__(self):
    self.output_triplets = np.array([
      [[1,1],[2,2],[5,5]],
      [[1,1],[5,5],[2,2]],
      [[1,1],[2,2],[5,5]],
      [[1,1],[5,5],[2,2]],
      [[1,1],[5,5],[5,5]]])
    print(type(self.output_triplets), 
          self.output_triplets.shape,
          self.output_triplets[0])
    #run test
    self.test_loss()
    self.test_HingeLoss()

  def test_loss(self):
    output_triplets = tf.cast(self.output_triplets, tf.float32)
    anchors, positives, negatives = tf.split(output_triplets,3,axis=1)
    pos_dist = tf.square(tf.subtract(anchors,positives), name="pos_dist")
    neg_dist = tf.square(tf.subtract(anchors,negatives), name="neg_dist")
    hinge_dist = tf.maximum(pos_dist - neg_dist + 1.0, 0.0)
    # hinge_dist = tf.reduce_sum(hinge_dist, 2)
    hinge_loss = tf.reduce_mean(hinge_dist)
    with tf.Session() as sess:
      results = sess.run((pos_dist, neg_dist, hinge_dist, hinge_loss))
      for i,result in enumerate(results):
        print("results[%d]:"%i,result, result.shape)
    assert results[3].astype(str)=='6.6'

  def test_HingeLoss(self):
    loss_fn = HingeLoss()
    loss = loss_fn.calculate_loss(self.output_triplets)
    with tf.Session() as sess:
      loss_val = sess.run(loss)
      print(loss_val, loss_val.shape)
      assert loss_val.astype(str)=='6.6'


if __name__ == "__main__":
    test_losses()