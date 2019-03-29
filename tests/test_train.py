# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from tensorflow import logging

import models
import inputs
import losses
from parse_data import lookup
from utils import find_class_by_name
from train import build_pipe_graph
from train import build_graph
from train import Trainer


logging.set_verbosity(logging.DEBUG)

@DeprecationWarning
def test_build_pipe_graph():
  triplets, _ = get_triplets(watch_file="watched_guids.txt",
                             feature_file="visual_features.txt")
  build_pipe_graph(triplets=triplets,
                   pipe=inputs.TripletPipe(),
                   batch_size=2)
  
  guid_triplets = tf.get_collection("guid_triplets")[0]
  with tf.Session() as sess:
    guid_triplets_val = sess.run(guid_triplets)
    print(guid_triplets_val)


def test_build_graph():
  train_dir="."
  model = find_class_by_name("VENet", [models])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  optimizer_class = find_class_by_name("AdamOptimizer", [tf.train])
  
  triplets, features = get_triplets(watch_file="watched_guids.txt",
                             feature_file="visual_features.txt")
  batch_size = 2
  num_epochs = None
  build_pipe_graph(triplets=triplets,
                   pipe=inputs.TripletPipe(),
                   batch_size=batch_size,
                   num_epochs=num_epochs)
  guid_triplets = tf.get_collection("guid_triplets")[0]

  input_triplets = tf.placeholder(tf.float32, name="input_triplets")
  build_graph(input_triplets=input_triplets,
              model=models.VENet(),
              output_size=256,
              loss_fn=losses.HingeLoss(),
              base_learning_rate=0.01,
              learning_rate_decay_examples=1000000,
              learning_rate_decay=0.95,
              optimizer_class=tf.train.AdamOptimizer,
              clip_gradient_norm=1.0,
              regularization_penalty=1)
  global_step = tf.train.get_or_create_global_step()
  output_batch = tf.get_collection("output_batch")[0]
  loss = tf.get_collection("loss")[0]
  train_op = tf.get_collection("train_op")[0]
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)
    for i in range(3):
      guid_triplets_val = sess.run(guid_triplets)
      input_triplets_val = lookup(guid_triplets_val, features)
      
      _, global_step_val, output_batch_val, loss_val = sess.run(
          [train_op, global_step, output_batch, loss],
          feed_dict={input_triplets: input_triplets_val})
      assert global_step_val == i+1
      assert output_batch_val.shape==(batch_size, 3, 256)
      print(i+1, loss_val)


def test_Trainer():
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 使用第 2 块GPU
  # original_data = '/home/wengjy1/cdml/tests/'
  original_data = None

  logging.info("Tensorflow version: %s.",tf.__version__)
  checkpoint_dir = "/home/wengjy1/checkpoints/"
  pipe = inputs.MPTripletPipe(original_data = original_data,
                              triplet_file_patten='/home/wengjy1/cdml/tests/*.triplet',
                              feature_file="/home/wengjy1/tests/cdml/features.txt",
                              debug=False)

  model = find_class_by_name("VENet", [models])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  optimizer_class = find_class_by_name("AdamOptimizer", [tf.train])
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
  config.gpu_options.allow_growth=True
  trainer = Trainer(pipe=pipe,
                    num_epochs=10,
                    batch_size=100,
                    wait_times=10,
                    model=model,
                    loss_fn=loss_fn,
                    checkpoint_dir=checkpoint_dir,
                    optimizer_class=optimizer_class,
                    config=config,
                    last_step=None,
                    debug=False)
  trainer.run()

if __name__ == "__main__":
  # test_build_graph()
  test_Trainer()
