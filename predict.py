# -*- coding:utf-8 -*-
""" Generates an output npy file containing predictions of
the model over a set of video features.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import logging
from online_data import read_features_npy
from utils import get_latest_folder

logging.set_verbosity(tf.logging.INFO)


class Prediction():
  def __init__(self, sess=None, ckpt_dir=None, config=None, device_name=None):
    self.sess = sess
    self.ckpt_dir = ckpt_dir
    self.config = config
    self.device_name = device_name
    if not sess:
      # 如果没有传入 session，根据 ckpt_dir 和 config 载入 session
      logging.info(str(self.ckpt_dir))
      checkpoint = tf.train.latest_checkpoint(ckpt_dir)
      if not checkpoint:
        logging.error("Prediction __init__. Load checkpoint failed: "+str(ckpt_dir))
      meta_graph = checkpoint + ".meta"
      if not os.path.exists(meta_graph):
        raise IOError("Prediction __init__ Cannot find %s." % checkpoint)
      with tf.device(device_name):
        loader = tf.train.import_meta_graph(meta_graph, clear_devices=True)
      self.sess = tf.Session(config=config)
      self.sess.graph.finalize()
      loader.restore(self.sess, checkpoint)
    
    # Get the tensors by their variable name
    graph = tf.get_default_graph()
    self.input_batch = graph.get_operation_by_name('input_batch').outputs[0]
    self.output_batch = graph.get_operation_by_name('VENet/model_output').outputs[0]
    logging.info('Prediction __init__ Load predictor!')

  def predict(self, input_batch_np):
    output_batch_np = self.sess.run(self.output_batch, feed_dict={self.input_batch: input_batch_np})
    return output_batch_np

  def run_features(self, features, output_dir, batch_size,suffix=''):
    logging.info('Predicting features...')

    steps = int(features.shape[0]/batch_size)
    tail_size = features.shape[0] - batch_size * steps
    output = []
    for step in range(steps):
      output_batch_np = self.predict(features[step:step+batch_size])
      output_batch_list = output_batch_np.tolist()
      output.extend(output_batch_list)
    if tail_size:
      output_batch_np = self.predict(features[features.shape[0]-tail_size:])
      output_batch_list = output_batch_np.tolist()
      output.extend(output_batch_list)

    output_np = np.asarray(output, np.float32)
    save_dir = os.path.join(output_dir,"output"+suffix+".npy")
    np.save(save_dir, output_np)
    print(output_np.shape, output_np[-1])
    logging.info('Saved'+save_dir)


if __name__ == "__main__":
  train_dir = "/data/wengjy1/train_dir/"
  ckpts_dir = train_dir+"checkpoints/"
  ckpt_dir = get_latest_folder(ckpts_dir,nst_latest=1)
  batch_size = 100000 

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth=True

  predictor = Prediction(ckpt_dir=ckpt_dir, config=config, device_name=None)
  features = read_features_npy(os.path.join(train_dir,"features.npy"))
  predictor.run_features(features=features, output_dir=ckpt_dir, batch_size=batch_size)
