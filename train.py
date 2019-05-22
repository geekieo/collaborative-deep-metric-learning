# -*- coding: utf-8 -*-
import time
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import logging
from tensorflow import flags
# from tensorflow.python.client import device_lib
import numpy as np
# import shutil
import traceback

import losses
import inputs
import models
from predict import Prediction
from evaluate import Evaluation
from online_data import load_cowatches
from utils import find_class_by_name
from utils import get_local_time
# from utils import get_latest_folder

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS

flags.DEFINE_string("train_dir", "/data/wengjy1/cdml_1_unique",
    "训练文件根目录，包括验证集和测试集")
flags.DEFINE_string("checkpoint_dir", "/data/wengjy1/cdml_1_unique/checkpoints",
    "存储每次训练产生的模型文件，包含 tensorboard")

def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.
  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.
  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, max_norm)
        grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = tf.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars


def calc_var(triplets, name=None):
  with tf.name_scope(name):
    mean = tf.reduce_mean(triplets,axis=[0,1])
    delta = triplets-mean
    return tf.reduce_mean(delta**2)


def build_graph(input_batch,
                model,
                output_size=256,
                loss_fn=losses.HingeLoss(),
                base_learning_rate=0.01,
                learning_rate_decay_examples=100000,
                learning_rate_decay=0.96,
                margin = 0.8,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    input_batch: tf.placehoder. tf.float32. shape(batch_size, 1500)
    loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.

  """
  #create_model loss train_op add_to_collection
  result = model.create_model(input_batch, output_size)
  
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(
    base_learning_rate,
    global_step,
    learning_rate_decay_examples,
    learning_rate_decay,
    staircase=True)

  if optimizer_class.__name__=="MomentumOptimizer":
    optimizer = optimizer_class(learning_rate, momentum=0.9,name='Momentum',use_nesterov=True)
  else:
    optimizer = optimizer_class(learning_rate)

  try:
    # In adagrad and gradient descent learning_rate is self._learning_rate.
    final_learning_rate = optimizer._learning_rate
  except:
    # In adam learning_rate is self._lr
    final_learning_rate = optimizer._lr

  output_batch = result["l2_norm"]  # shape: (-1,output_size)
  output_triplets = tf.reshape(output_batch,(-1,3,output_size))

  loss_result = loss_fn.calculate_loss(output_triplets, margin=margin)
  loss = loss_result['hinge_loss']

  reg_loss = tf.constant(0.0)
  reg_losses = tf.losses.get_regularization_losses()
  if reg_losses:
    reg_loss += tf.add_n(reg_losses)

  # Incorporate the L2 weight penalties etc.
  final_loss = regularization_penalty * reg_loss + loss

  gradients = optimizer.compute_gradients(final_loss,
              colocate_gradients_with_ops=False)
  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
      gradients = clip_gradient_norms(gradients, clip_gradient_norm)
  train_op = optimizer.apply_gradients(gradients, global_step=global_step)
  
  # train_op = optimizer.minimize(final_loss, global_step=global_step)

  # 可分性好的 embeddings， 那么其方差应该是偏大的
  variance = calc_var(output_triplets, "variance")

  # summary
  with tf.name_scope('build_graph'):
    # 模型输出
    tf.summary.scalar("loss", loss)
    if regularization_penalty != 0:
      tf.summary.scalar("reg_loss", reg_loss)
    tf.summary.scalar("variance", variance)
    tf.summary.scalar('final_learning_rate', final_learning_rate)

  tf.add_to_collection("output_batch", output_triplets)  
  tf.add_to_collection("loss", loss)
  tf.add_to_collection("train_op", train_op)
  #debug
  tf.add_to_collection("layer_1", result["layer_1"])
  tf.add_to_collection("layer_2", result["layer_2"])
  # tf.add_to_collection("gradients", gradients) 
  tf.add_to_collection("anchors",loss_result["anchors"])
  tf.add_to_collection("positives",loss_result["positives"])
  tf.add_to_collection("negatives",loss_result["negatives"])
  tf.add_to_collection("pos_dist",loss_result["pos_dist"])
  tf.add_to_collection("neg_dist",loss_result["neg_dist"])
  tf.add_to_collection("hinge_dist", loss_result["hinge_dist"])
  tf.add_to_collection("hinge_loss",loss_result['hinge_loss'])
  tf.add_to_collection("final_loss",final_loss)


class Trainer():

  def __init__(self, pipe, num_epochs, batch_size, model, loss_fn, learning_rate, margin,
               checkpoint_dir, optimizer_class, config, eval_cowatches, test_cowatches,
               best_eval_dist=1.0, require_improve_num=10,loglevel=tf.logging.INFO):
    # self.is_master = (task.type == "master" and task.index == 0)
    # self.is_master = True 
    self.pipe = pipe
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.model = model
    self.loss_fn = loss_fn
    self.learning_rate = learning_rate
    self.margin = margin
    # self.checkpoint_dir = os.path.join(checkpoint_dir, 
    #                       model.__class__.__name__+"_"+get_local_time())
    self.checkpoint_dir = checkpoint_dir
    self.optimizer_class = optimizer_class
    self.config = config

    # 验证早停及模型选取依据
    self.total_eval_num = 0
    self.last_improve_num = 0
    self.best_eval_dist = best_eval_dist
    self.eval_dist = 0.0
    self.require_improve_num = require_improve_num # 如果几次验证没有改进，停止迭代
    # 准备验证对象
    self.evaluater = Evaluation(inputs.FEATURES, eval_cowatches)
    # 准备测试对象
    self.tester = Evaluation(inputs.FEATURES, test_cowatches)

    logging.set_verbosity(loglevel)

  def _build_model(self,input_batch):
    """Find the model and build the graph."""
    build_graph(input_batch=input_batch,
                model=self.model,
                output_size=256,
                loss_fn=self.loss_fn,
                base_learning_rate=self.learning_rate,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.96,
                margin = self.margin,
                optimizer_class=self.optimizer_class,
                clip_gradient_norm=0,
                regularization_penalty=0)

  def _eval(self, predictor, saver, sess, global_step_np, summary_writer, check_stop_step):
    self.total_eval_num += 1
    try:
      if self.evaluater.features is None:
        logging.error('Train.run evaluater.features is None')
        raise
      eval_embeddings = predictor.run_features(self.evaluater.features, batch_size=10000)
      self.eval_dist = self.evaluater.mean_dist(eval_embeddings, self.evaluater.cowatches)
      if global_step_np <= check_stop_step:
        logging.info("Eval "+str(self.total_eval_num)+" | best_eval_dist: "+
            str(self.best_eval_dist)+" eval_dist: "+str(self.eval_dist)+". Before check stop.")
      elif self.eval_dist < self.best_eval_dist:
        logging.info("Eval "+str(self.total_eval_num)+" | best_eval_dist: "+
            str(self.best_eval_dist)+" > eval_dist: "+str(self.eval_dist)+". Save ckpt.")
        self.best_eval_dist = self.eval_dist
        saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_np)
        self.last_improve_num = self.total_eval_num
      else:
        logging.info("Eval "+str(self.total_eval_num)+" | best_eval_dist: "+
            str(self.best_eval_dist)+" < eval_dist: "+str(self.eval_dist)+
            ". From the last improvement: "+str(self.total_eval_num-self.last_improve_num))
      summary_eval = tf.Summary(value=[
        tf.Summary.Value(tag="eval/eval_dist", simple_value=self.eval_dist), 
        tf.Summary.Value(tag="eval/best_eval_dist", simple_value=self.best_eval_dist)])
      summary_writer.add_summary(summary_eval, global_step_np)

      # test_dist = _test(predictor)
      # summary_test = tf.Summary(value=[
      #     tf.Summary.Value(tag="eval/test_dist", simple_value=test_dist)])
      # summary_writer.add_summary(summary_test, global_step_np)
    except Exception as e:
      logging.error("Train._eval "+str(e))

  def _test(self, predictor):
    if self.tester.features is None:
      logging.error('Train.run tester.features is None')
    test_embeddings = predictor.run_features(self.tester.features, batch_size=50000)
    return self.evaluater.mean_dist(test_embeddings, self.tester.cowatches)

  # def _copy_ckpt(self, ckpt, output_dir):
  #   data = ckpt+".data-00000-of-00001"
  #   index = ckpt + ".index"
  #   meta = ckpt + ".meta"
  #   try:
  #     shutil.copyfile(data, output_dir + "/"+get_local_time()+".data-00000-of-00001")
  #     shutil.copyfile(index, output_dir + "/"+get_local_time()+".index")
  #     shutil.copyfile(meta, output_dir + "/"+get_local_time()+".meta")
  #   except Exception as e:
  #     logging.error(traceback.format_exc())


  # def _deploy_best_ckpt(self, checkpoints_dir, output_dir):
  #   ckpt_dir = get_latest_folder(checkpoints_dir, nst_latest=1)
  #   today_ckpt = tf.train.latest_checkpoint(ckpt_dir)
  #   try:
  #     ckpt_dir = get_latest_folder(checkpoints_dir, nst_latest=2)
  #     yesterday_ckpt = tf.train.latest_checkpoint(ckpt_dir)
  #   except Exception as e:
  #     logging.info('Got no yesterday\'s ckpt. Deploy today\'s ckpt to ' + output_dir)
  #     _copy_ckpt(today_ckpt, output_dir)
  #     break
    


  def run(self):

    self.pipe.create_pipe(self.num_epochs, self.batch_size)

    logging.info("Building model graph.")
    input_batch = tf.placeholder(tf.float32, shape=(None,1500), name="input_batch")
    self._build_model(input_batch)

    global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection("loss")[0]
    output_batch = tf.get_collection("output_batch")[0]
    train_op = tf.get_collection("train_op")[0]
    init_op = tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=1)   # 仅保存1个ckpt, 验证效果最好的ckpt
        
    logging.debug("Train.pipe.get_batch"+str(self.pipe.get_batch().shape))
    logging.info("Starting session.")
    with tf.Session(config=self.config) as sess:
      sess.run(init_op)
      summary_writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)
      predictor = Prediction(sess=sess)

      global_step_np = 0
      check_stop_step = 30000
      # 迭代拟合
      while True:
        try:
          fetch_start_time = time.time()
          input_triplets_np = self.pipe.get_batch()
          if input_triplets_np is None:
            if self.eval_dist < self.best_eval_dist:
              logging.info("Didn't check stop in Eval "+str(self.total_eval_num)+" | best_eval_dist: "+
                  str(self.best_eval_dist)+" > eval_dist: "+str(self.eval_dist)+". Save ckpt in the end.")
              saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_np)
            break
          if self.total_eval_num - self.last_improve_num +1 > self.require_improve_num and global_step_np > check_stop_step:
            logging.info("early stop")
            break
          if not input_triplets_np.shape[1:] == (3,1500):
            continue
          # print('input_triplets_np.shape: ',input_triplets_np.shape) #debug
          input_batch_np = np.reshape(input_triplets_np, (-1,input_triplets_np.shape[-1])) # 3-D to 2-D
          fetch_time = time.time() - fetch_start_time

          batch_start_time = time.time()
          _, global_step_np, loss_np = sess.run([train_op, global_step, loss],
                feed_dict={input_batch: input_batch_np})
          trian_time = time.time() - batch_start_time
          if global_step_np % 40 == 0:
            logging.debug("Step " + str(global_step_np) + " | Loss: " + ("%.8f" % loss_np) +
                " | Time: fetch: " + ("%.4f" % fetch_time) + "sec"
                " train: " + ("%.4f" % trian_time)+"sec")
          if global_step_np % 400 == 0:
            self._eval(predictor, saver, sess, global_step_np, summary_writer, check_stop_step)
            summary_str = sess.run(summary_op, feed_dict={input_batch: input_batch_np})
            summary_writer.add_summary(summary_str, global_step_np)
          # if global_step_np == 5000:
          #   saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_np)
          #   break
        except Exception as e:
          logging.error("Train.run "+str(e)) 
      # TODO tests and backup best model
      summary_writer.close()
      self.pipe.__del__()
      logging.info("Exited training loop.")


def main(args):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 使用第 2 块GPU
  # TODO Prepare distributed arguments here. 
  logging.info("Tensorflow version: %s.",tf.__version__)
  checkpoint_dir = FLAGS.checkpoint_dir
  pipe = inputs.MPTripletPipe(cowatch_file_patten = FLAGS.train_dir + "/*.train",
                              feature_file = FLAGS.train_dir + "/features.npy",
                              wait_times=20)
  eval_cowatches =  load_cowatches(FLAGS.train_dir + "/cowatches.eval")
  test_cowatches =  load_cowatches(FLAGS.train_dir + "/cowatches.test")

  model = find_class_by_name("VENet", [models])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  # optimizer_class = find_class_by_name("AdamOptimizer", [tf.train])
  optimizer_class = tf.contrib.opt.LARSOptimizer
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
  config.gpu_options.allow_growth=True

  trainer = Trainer(pipe=pipe,
                    num_epochs=5,
                    batch_size=1024,
                    model=model,
                    loss_fn=loss_fn,
                    learning_rate=1.0,
                    margin=0.8,
                    checkpoint_dir=checkpoint_dir,
                    optimizer_class=optimizer_class,
                    config=config,
                    eval_cowatches=eval_cowatches,
                    test_cowatches=test_cowatches,
                    best_eval_dist=1.0,
                    require_improve_num=100,
                    loglevel=tf.logging.INFO)
  # TODO model test and backup. new model vs last model
  
  trainer.run()


if __name__ == "__main__":
  tf.app.run()
