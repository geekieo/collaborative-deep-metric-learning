# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from imitation_data import gen_features
from inputs import TripletPipe


class test_TripletPipe():

  def __init__(self):
    self.test_create_pipe()
    
  def test_create_pipe(self):
    input_shape = (3, 2, 3)
    features = gen_features(num_feature=input_shape[0]*input_shape[1],
                            feature_size=input_shape[2])
    features = np.reshape(features,input_shape)
    pipe = TripletPipe()
    batch_size = 5
    input_tensor = pipe.create_pipe(features, batch_size=batch_size, num_epochs=None)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      input_val = sess.run(input_tensor)
      print(input_val, input_val.shape)
      assert input_val.shape == (batch_size, input_shape[1], input_shape[2])

if __name__ == "__main__":
  test_TripletPipe()