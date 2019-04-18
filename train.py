# -*- coding: utf-8 -*-
import time
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import logging
# from tensorflow.python.client import device_lib
import numpy as np

import losses
import inputs
import models
from predict import Prediction
from utils import find_class_by_name
from utils import get_local_time

logging.set_verbosity(logging.DEBUG)


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

@DeprecationWarning
def build_pipe_graph(triplets,
                     pipe,
                     batch_size=100,
                     num_epochs=None,
                     num_readers=1):
  """run pipe in CPU memory
  Args:
    triplets: 3-D list. guid triplets
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
    num_readers: How many threads to use for I/O operations.
  """
  triplets_iter = pipe.create_pipe(triplets, batch_size=batch_size, num_epochs=num_epochs)
  guid_triplets = triplets_iter.get_next()
  tf.add_to_collection("guid_triplets", guid_triplets)


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

  loss_result = loss_fn.calculate_loss(output_triplets, margin=0.8)
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
  # # 模型权重
  # for variable in slim.get_model_variables():
  #   tf.summary.histogram(variable.op.name, variable)


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

  def __init__(self, pipe, num_epochs, batch_size, wait_times, model, loss_fn, 
               checkpoints_dir, optimizer_class, config,
               last_step=None, debug=False):
    # self.is_master = (task.type == "master" and task.index == 0)
    # self.is_master = True 
    self.pipe = pipe
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.wait_times = wait_times
    self.model = model
    self.loss_fn = loss_fn
    self.checkpoint_dir = os.path.join(checkpoints_dir, 
                          model.__class__.__name__+"_"+get_local_time())
    self.optimizer_class = optimizer_class
    self.config = config
    self.last_step = last_step
    self.debug = debug

  @DeprecationWarning
  def build_pipe(self):
    """build the pipe graph """
    build_pipe_graph(triplets=self.triplets,
                     pipe=self.pipe,
                     batch_size=self.batch_size,
                     num_epochs=self.num_epochs)

  def build_model(self,input_batch):
    """Find the model and build the graph."""
    build_graph(input_batch=input_batch,
                model=self.model,
                output_size=256,
                loss_fn=self.loss_fn,
                base_learning_rate=1.0,
                learning_rate_decay_examples=100000,
                learning_rate_decay=0.96,
                optimizer_class=self.optimizer_class,
                clip_gradient_norm=0,
                regularization_penalty=0)

  def run(self):

    self.pipe.create_pipe(self.num_epochs, self.batch_size)

    # with tf.device('/cpu:0'):
    logging.info("Building model graph.")
    input_batch = tf.placeholder(tf.float32, shape=(None,1500), name="input_batch")
    self.build_model(input_batch)


    global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection("loss")[0]
    output_batch = tf.get_collection("output_batch")[0]
    train_op = tf.get_collection("train_op")[0]
    init_op = tf.global_variables_initializer()

    # debug
    layer_1 = tf.get_collection("layer_1")[0]
    layer_2 = tf.get_collection("layer_2")[0]
    anchors = tf.get_collection("anchors")[0]
    positives = tf.get_collection("positives")[0]
    negatives = tf.get_collection("negatives")[0]
    pos_dist = tf.get_collection("pos_dist")[0]
    neg_dist = tf.get_collection("neg_dist")[0]
    hinge_dist = tf.get_collection("hinge_dist")[0]
    hinge_loss = tf.get_collection("hinge_loss")[0]
    final_loss = tf.get_collection("final_loss")[0]
    # gradients = tf.get_collection("gradients")[0]
    
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    logging.info("Starting session.")
    with tf.Session(config=self.config) as sess:
      sess.run(init_op)
      train_writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)
      summary_np=None  #暂存上个循环的 summary，以在循环结束时写入最后一次成功运行的 summary
      evaluator = Prediction(sess=sess, device_name=None)
      while True:
        try:
          fetch_start_time = time.time()
          input_triplets_np = self.pipe.get_batch(self.wait_times)
          if input_triplets_np is None:
            # summary save model
            train_writer.add_summary(summary_np, global_step_np)
            saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_np)
            evaluator.run_features(inputs.FEATURES, output_dir=self.checkpoint_dir,
                                   batch_size=50000, suffix=str(global_step_np))
            logging.info('Done training. Pipe end! Add summary. Save checkpoint.')
            break
          if not input_triplets_np.shape == (self.batch_size,3,1500):
            continue
          # print('input_triplets_np.shape: ',input_triplets_np.shape)
          if self.debug:
            logging.debug(type(input_triplets_np)+input_triplets_np.shape+input_triplets_np.dtype)
          input_batch_np = np.reshape(input_triplets_np, (-1,input_triplets_np.shape[-1])) # 3-D to 2-D
          fetch_time = time.time() - fetch_start_time

          batch_start_time = time.time()
          _, global_step_np, loss_np, summary_np= sess.run(
              [train_op, global_step, loss,summary_op],
              feed_dict={input_batch: input_batch_np})

          # _, global_step_np, loss_np, summary_np, input_batch_np, layer_1_np, layer_2_np, output_batch_np, anchors_np, positives_np, negatives_np,pos_dist_np,neg_dist_np,hinge_dist_np,hinge_loss_np, final_loss_np= sess.run(
          #     [train_op, global_step, loss, summary_op,input_batch, layer_1, layer_2, output_batch, anchors, positives, negatives,pos_dist,neg_dist,hinge_dist, hinge_loss, final_loss],
          #     feed_dict={input_batch: input_batch_np})
          # print('input_batch_np',input_batch_np.shape,input_batch_np,
          #     'layer_1_np',layer_1_np.shape,layer_1_np,
          #     'layer_2_np',layer_2_np.shape,layer_2_np,
          #     'output_batch_np',output_batch_np.shape,output_batch_np, 
          #     'anchors_np',anchors_np.shape,anchors_np, 
          #     'positives_np',positives_np.shape, positives_np, 
          #     'negatives_np',negatives_np.shape, negatives_np, 
          #     'pos_dist_np',pos_dist_np.shape, pos_dist_np, 
          #     'neg_dist_np',neg_dist_np.shape, neg_dist_np, 
          #     'hinge_dist_np',hinge_dist_np.shape, hinge_dist_np, 
          #     'hinge_loss_np',hinge_loss_np.shape, hinge_loss_np, 
          #     'final_loss_np',final_loss_np.shape, final_loss_np,
          #     sep='\n')

          trian_time = time.time() - batch_start_time

          if global_step_np % 200 == 0:
            logging.info("Step " + str(global_step_np) + " | Loss: " + ("%.8f" % loss_np) +
                " | Time: fetch: " + ("%.4f" % fetch_time) + "sec"
                " train: " + ("%.4f" % trian_time)+"sec")
          if global_step_np % 2000 == 0:
            train_writer.add_summary(summary_np, global_step_np)
            logging.info("add summary")
          if global_step_np % 20000 == 0:
            saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_np)
            logging.info("save checkpoint")
            evaluator.run_features(inputs.FEATURES, output_dir=self.checkpoint_dir,
                                   batch_size=50000, suffix=str(global_step_np))
            pass

        except Exception as e:
          logging.error(str(e)) 
      self.pipe.__del__()
      logging.info("Exited training loop.")


def main(args):
  # TODO Prepare distributed arguments here. 
  logging.info("Tensorflow version: %s.",tf.__version__)
  train_dir = "/data/wengjy1/cdml_1"  # NOTE 路径是 data
  checkpoints_dir = train_dir+"/checkpoints/"
  pipe = inputs.MPTripletPipe(cowatch_file_patten = train_dir + "/*.cowatch",
                                feature_file = train_dir + "/features.npy")
  model = find_class_by_name("VENet", [models])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  optimizer_class = find_class_by_name("MomentumOptimizer", [tf.train])
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
  config.gpu_options.allow_growth=True
  trainer = Trainer(pipe=pipe,
                    num_epochs=6,
                    batch_size=1024,
                    wait_times=20,
                    model=model,
                    loss_fn=loss_fn,
                    checkpoints_dir=checkpoints_dir,
                    optimizer_class=optimizer_class,
                    config=config,
                    last_step=None,
                    debug=False)
  trainer.run()

if __name__ == "__main__":
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 使用第 2 块GPU
  tf.app.run()
