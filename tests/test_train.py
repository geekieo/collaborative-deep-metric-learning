# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from tensorflow import logging

import models
import inputs
import losses
from utils import find_class_by_name
from train import build_graph
from train import Trainer


logging.set_verbosity(logging.DEBUG)


def test_build_graph():
  train_dir="."
  model = find_class_by_name("VENet", [models])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  optimizer_class = find_class_by_name("AdamOptimizer", [tf.train])
  data = np.array([
        [[1,1],[2,2],[5,5]],
        [[1,1],[5,5],[2,2]],
        [[1,1],[2,2],[5,5]],
        [[1,1],[5,5],[2,2]],
        [[1,1],[5,5],[5,5]]])

  build_graph(pipe=inputs.TripletPipe(),
              data=data,
              model=models.VENet(),
              output_size=256,
              loss_fn=losses.HingeLoss(),
              batch_size=10,
              base_learning_rate=0.01,
              learning_rate_decay_examples=1000000,
              learning_rate_decay=0.95,
              optimizer_class=tf.train.AdamOptimizer,
              clip_gradient_norm=1.0,
              regularization_penalty=1,
              num_epochs=None,
              num_readers=1)
  global_step = tf.get_collection("global_step")[0]
  input_triplets = tf.get_collection("input_batch")[0]
  output_triplets = tf.get_collection("output_batch")[0]
  loss = tf.get_collection("loss")[0]
  train_op = tf.get_collection("train_op")[0]
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    global_step_val, input_val, output_val, loss_val,train_op_val = sess.run(
      [global_step, input_triplets, output_triplets, loss, train_op])
    assert global_step_val == 1
    assert input_val.shape==(10, 3, 2)
    assert output_val.shape==(10, 3, 256)
    assert loss_val==1.0
    assert train_op_val==None

def test_Trainer():
  # from imitation_data import gen_triplets
  # triplets = gen_triplets(batch_size=3000,feature_size=1500)

  from online_data import get_triplets
  triplets = get_triplets(watch_file="watched_guids.txt",
                          feature_file="features.txt",
                          return_features=True)
  triplets = np.array(triplets, dtype=np.float32)
  print(triplets.shape)
  logging.info("Tensorflow version: %s.",tf.__version__)
  checkpoint_dir = "/Checkpoints/"
  model = find_class_by_name("VENet", [models])()
  pipe = find_class_by_name("TripletPipe", [inputs])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  optimizer_class = find_class_by_name("AdamOptimizer", [tf.train])
  # cluster = None
  # task_data = {"type": "master", "index": 0}
  # task = type("TaskSpec", (object,), task_data)
  trainer = Trainer(checkpoint_dir=checkpoint_dir,
                    data=triplets,
                    model=model,
                    pipe=pipe,
                    loss_fn=loss_fn,
                    optimizer_class=optimizer_class,
                    batch_size=100,
                    num_epochs=None)
  trainer.run() 


if __name__ == "__main__":
  # test_build_graph()
  test_Trainer()