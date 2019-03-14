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
    input_shape = (5, 3, 2)
    triplets = gen_features(num_feature=input_shape[0]*input_shape[1],
                            feature_size=input_shape[2])
    triplets = np.reshape(triplets,input_shape)
    pipe = TripletPipe()
    batch_size = 3
    input_iter = pipe.create_pipe(data=triplets, batch_size=batch_size, num_epochs=None)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      input_val = sess.run(input_iter.get_next())
      print(input_val, input_val.shape)
      assert input_val.shape == (batch_size, input_shape[1], input_shape[2])
      input_val = sess.run(input_iter.get_next())
      print(input_val, input_val.shape)
      assert input_val.shape == (batch_size, input_shape[1], input_shape[2])
    

if __name__ == "__main__":
  test_TripletPipe()