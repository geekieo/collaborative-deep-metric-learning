# -*- coding:utf-8 -*-
""" Generates an output npy file containing predictions of
the model over a set of video features.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import logging
from online_data import read_features_npy

logging.set_verbosity(tf.logging.INFO)


class Prediction():
  def __init__(self, ckpt_dir, config, device_name=None):
    self.ckpt_dir = ckpt_dir
    
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if not checkpoint:
      logging.error("Load checkpoint failed: "+str(ckpt_dir))
    meta_graph = checkpoint + ".meta"
    if not os.path.exists(meta_graph):
      raise IOError("Cannot find %s." % checkpoint)
    with tf.device(device_name):
      loader = tf.train.import_meta_graph(meta_graph, clear_devices=True)

    self.sess = tf.Session(config=config)
    self.sess.graph.finalize()
    graph = tf.get_default_graph()
    loader.restore(self.sess, checkpoint)
    # Get the tensors by their variable name
    self.input_batch = graph.get_operation_by_name('input_batch').outputs[0]
    self.output_batch = graph.get_operation_by_name('VENet/l2_normalize').outputs[0]
    
    logging.info('Load predictor!')

  def predict(self, input_batch_np):
    output_batch_np = self.sess.run(self.output_batch, feed_dict={self.input_batch: input_batch_np})
    return output_batch_np

  def run_features(self, train_dir, batch_size):
    feature_file = os.path.join(train_dir,"features.npy")
    features = read_features_npy(feature_file)

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
    np.save(self.ckpt_dir+"output.npy", output_np)
    print(output_np.shape, output_np[-1])
    logging.info('Saved output.npy')


def get_latest_folder(checkpoints_dir, nst_latest=1):
  """获取目录文件夹，根据创建时间排序，
  返回第 nst_latest 新的文件夹路径
  """
  files = os.listdir(checkpoints_dir)
  folders = []
  for file in files:
    path = os.path.join(checkpoints_dir, file)
    # print(path, os.path.getctime(path))
    if os.path.isdir(path):
      folders.append(path)
  folders.sort(key=lambda folder: os.path.getmtime(folder))
  return folders[-nst_latest]

if __name__ == "__main__":
  train_dir = "/data/wengjy1/train_dir/"
  ckpts_dir = train_dir+"checkpoints/"
  ckpt_dir = get_latest_folder(ckpts_dir,nst_latest=1)
  batch_size = 2000 

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth=True

  predictor = Prediction(checkpoint, config, device_name=None)
  predictor.run_features(train_dir, ckpt_dir, batch_size)
