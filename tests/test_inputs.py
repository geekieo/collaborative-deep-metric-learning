# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from online_data import get_triplets
from parse_data import lookup
from inputs import TripletPipe


def test_TripletPipe():
  triplets, features = get_triplets(watch_file="watched_guids.txt",
                                    feature_file="features.txt")
  print("triplets type",type(triplets),type(triplets[0]),type(triplets[0][0]))
  print("triplets shape",len(triplets),len(triplets[0]),len(triplets[0][0]))
  # build graph
  pipe = TripletPipe(triplets)
  batch_size = 2
  triplets_iter = pipe.create_pipe(batch_size=batch_size, num_epochs=None)
  guid_triplets = triplets_iter.get_next()
  tf.add_to_collection("guid_triplets", guid_triplets)
  # use graph
  guid_triplets = tf.get_collection("guid_triplets")[0]
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    guid_triplets_val = sess.run(guid_triplets)
    guid_triplets_val = sess.run(guid_triplets)
    print(guid_triplets_val, guid_triplets_val.shape)
    assert guid_triplets_val.shape == (batch_size, 3)
    # trans guid to feature
    input_triplets = lookup(guid_triplets_val, features)
    print(input_triplets)
    print(input_triplets.shape)
    assert input_triplets.shape == (2, 3, 1500)

def test_


if __name__ == "__main__":
  # test_TripletPipe()