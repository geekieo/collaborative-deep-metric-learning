# -*- coding:utf-8 -*-
""" Generates an output npy file containing predictions of
the model over a set of video features.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import logging
from tensorflow import flags
import json
import traceback

from online_data import read_features_txt
from utils import get_latest_folder

logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 使用第2块GPU
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/data/wengjy1/cdml_1/checkpoints",
    "服务模型的目录，含备份模型")
flags.DEFINE_string("feature_file", "/data/wengjy1/1905cdml/features",
    "待预测的原始向量文件")
flags.DEFINE_string("output_dir", "/data/wengjy1/cdml_1/checkpoints",
    "模型输出向量的保存路径")
flags.DEFINE_integer("batch_size",300000,
    "每次预测的向量数")

class Prediction():
  def __init__(self, sess=None, ckpt=None, config=None, loglevel=tf.logging.INFO):
    logging.set_verbosity(loglevel)
    self.sess = sess
    self.ckpt = ckpt
    self.config = config
    if not sess:
      # 如果没有传入 session，根据 ckpt 和 config 载入 session
      logging.info(str(self.ckpt))
      meta_graph = self.ckpt + ".meta"
      if not os.path.exists(meta_graph):
        raise IOError("Prediction __init__ Cannot find %s" % self.ckpt)
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


def test_predict():
  ckpt_dir = FLAGS.model_dir
  logging.info("ckpt is "+ckpt_dir)
  ckpt = tf.train.latest_checkpoint(ckpt_dir)
  logging.info("predict read_features_txt reading ...")
  features, _, decode_map = read_features_txt(FLAGS.feature_file)
  logging.info("predict read_features_txt success!")
  # predict
  predictor = Prediction(ckpt=ckpt, config=config, loglevel=tf.logging.DEBUG)
  embeddings = predictor.run_features(features=features, batch_size=FLAGS.batch_size, output_dir=FLAGS.output_dir)
 
  print(features.shape)
  uni_embeddings = np.unique(embeddings, axis=0)
  print(uni_embeddings.shape)
  print((features.shape[0]-uni_embeddings.shape[0])/features.shape[0])  # Repetition rate


def main(args):
  # TODO logging.info("FLAGS")
  try:
    ckpt_dir = get_latest_folder(FLAGS.model_dir,nst_latest=1)
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if not os.path.exists(ckpt + ".meta"):
      ckpt_dir_2 = get_latest_folder(FLAGS.model_dir,nst_latest=2)
      ckpt_2 = tf.train.latest_checkpoint(ckpt_dir)
      ckpt = ckpt_2
    else:
      raise IOError("Prediction main Cannot find %s or %s" % ckpt, ckpt_2)
    logging.info("ckpt is "+ckpt)

    logging.info("predict read_features_txt reading ...")
    features, _, decode_map = read_features_txt(FLAGS.feature_file)
    logging.info("predict read_features_txt success!")
    # predict and write
    predictor = Prediction(ckpt=ckpt, config=config, loglevel=tf.logging.DEBUG)
    predictor.run_features(features=features, batch_size=FLAGS.batch_size, output_dir=FLAGS.output_dir)
    # write features and decode_map
    features_path = os.path.join(FLAGS.output_dir, '/features.npy')
    np.save(features_path, features)
    logging.info("predict write features.npy success!")
    with open(os.path.join(FLAGS.output_dir,'/decode_map.json'), 'w') as file:
      json.dump(decode_map, file, ensure_ascii=False)
    logging.info("predict write decode_map.json success!")
  except Exception as e:
    logging.error(traceback.format_exc())

if __name__ == "__main__":
  tf.app.run()