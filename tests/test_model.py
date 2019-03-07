# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np

from utils import find_class_by_name
import models
# from imitation_data import num_uid, num_guid, feature_size
from imitation_data import gen_features


class test_model():
  def __init__(self, model_name):
    self.model = find_class_by_name(model_name, [models])()
    # input 
    self.input_shape = (100, 3,1500)
    features = gen_features(num_feature=self.input_shape[0]*self.input_shape[1],
                            feature_size=self.input_shape[2])
    features_tensor = tf.constant(features)
    self.model_input = tf.reshape(features_tensor,self.input_shape)
    self.output_size = 256
    self.output_shape = (self.input_shape[0], self.input_shape[1], self.output_size)
    # run test
    self.test_create_model()

  def test_create_model(self):
    # inference a batch of feature
    result = self.model.create_model(self.model_input, self.output_size)
    output = result["output"]
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      result_val = sess.run(output)
    print(result_val.shape, result_val[0][0][0])
    assert result_val.shape == self.output_shape


def test_VENet():
  test_model("VENet")

if __name__ == "__main__":
    test_VENet()