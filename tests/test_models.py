# -*- coding: utf-8 -*-
'''
@Description: 
@Date: 2019-07-10 17:31:26
@Author: Weng Jingyu
'''
# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np

from utils import find_class_by_name
import models
import inputs
from imitation_data import gen_features


class test_model():
  def __init__(self, model_name):
    self.model = find_class_by_name(model_name, [models])()
    # input 
    self.input_shape = (200, 3,1500)
    features = gen_features(num_feature=self.input_shape[0]*self.input_shape[1],
                            feature_size=self.input_shape[2])
    features_tensor = tf.constant(features)
    triplets = tf.reshape(features_tensor,self.input_shape)
    # pipe input
    pipe = find_class_by_name("TripletPipe", [inputs])()  # pipe = inputs.TripletPipe()
    self.batch_size = 100
    input_iter = pipe.create_pipe(triplets, batch_size=self.batch_size, num_epochs=None)
    self.input_triplets = input_iter.get_next()
    # output
    self.output_size = 256
    self.output_shape = (self.batch_size, self.input_shape[1], self.output_size)
    # run test
    self.test_create_model()

  def test_create_model(self):
    # inference a batch of feature
    result = self.model.create_model(self.input_triplets, self.output_size)
    triplet_embed = result["output"]
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      triplet_np = sess.run(triplet_embed)
    print(triplet_np.shape, triplet_np[0][0][0])
    print(self.output_shape)
    assert triplet_np.shape == self.output_shape


def test_VENet():
  test_model("VeNet")

if __name__ == "__main__":
  test_VENet()