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
  def __init__(self, sess=None, ckpt=None, config=None, device_name=None, loglevel=tf.logging.INFO):
    logging.set_verbosity(loglevel)
    self.sess = sess
    self.ckpt = ckpt
    self.config = config
    self.device_name = device_name
    if not sess:
      # 如果没有传入 session，根据 ckpt 和 config 载入 session
      logging.info(str(self.ckpt))
      meta_graph = self.ckpt + ".meta"
      if not os.path.exists(meta_graph):
        raise IOError("Prediction __init__ Cannot find %s" % self.ckpt)
      with tf.device(device_name):
        loader = tf.train.import_meta_graph(meta_graph, clear_devices=True)
      self.sess = tf.Session()
      self.sess.graph.finalize()
      loader.restore(self.sess, self.ckpt)
    
    # Get the tensors by their variable name
    graph = tf.get_default_graph()
    self.input_batch = graph.get_operation_by_name('input_batch').outputs[0]
    self.output_batch = graph.get_operation_by_name('VENet/model_output').outputs[0]
    logging.info('Prediction __init__ Load predictor!')

  def predict(self, input_batch_np):
    output_batch_np = self.sess.run(self.output_batch, feed_dict={self.input_batch: input_batch_np})
    return output_batch_np

  def run_features(self, features, batch_size, output_dir='', suffix=''):
    logging.debug('Predicting features...')

    steps = int(features.shape[0]/batch_size)
    tail_size = features.shape[0] - batch_size * steps
    output = []
    for step in range(steps):
      output_batch_np = self.predict(features[step*batch_size:step*batch_size+batch_size])
      output_batch_list = output_batch_np.tolist()
      output.extend(output_batch_list)
    if tail_size:
      output_batch_np = self.predict(features[features.shape[0]-tail_size:])
      output_batch_list = output_batch_np.tolist()
      output.extend(output_batch_list)

    output_np = np.asarray(output, np.float32)
    logging.debug('Predict done.')
    if output_dir:
      try:
        save_dir = os.path.join(output_dir,"output"+suffix+".npy")
        np.save(save_dir, output_np)
        # print(output_np.shape, output_np[-1])
        logging.info('Saved to '+save_dir)
      except Exception as e:
        logging.error('Prediction.run_features save error'+str(e))
    else:
      logging.debug('No save.')
    return output_np


if __name__ == "__main__":
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 使用第1块GPU
  train_dir = "/data/wengjy1/cdml_1"  # NOTE 路径是 data
  checkpoints_dir = train_dir+"/checkpoints/"
  ckpt_dir = get_latest_folder(checkpoints_dir,nst_latest=1)
  ckpt = tf.train.latest_checkpoint(ckpt_dir)
  # ckpt = ckpt_dir+'/model.ckpt-800000'
  batch_size = 100000
  features = read_features_npy(train_dir+"/features.npy")

  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth=True

  predictor = Prediction(ckpt=ckpt, config=config, device_name=None, loglevel=tf.logging.DEBUG)
  embeddings = predictor.run_features(features=features, batch_size=batch_size, output_dir=ckpt_dir)

  print(features.shape)
  uni_embeddings = np.unique(embeddings, axis=0)
  print(uni_embeddings.shape)
  print((features.shape[0]-uni_embeddings.shape[0])/features.shape[0])  # Repetition rate