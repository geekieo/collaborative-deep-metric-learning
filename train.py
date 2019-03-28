# -*- coding: utf-8 -*-
import time
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import logging
# from tensorflow import flags
# from tensorflow.python.client import device_lib

import losses
import inputs
import models
from utils import find_class_by_name
from parse_data import lookup

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

def build_graph(input_triplets,
                model,
                output_size=256,
                loss_fn=losses.HingeLoss(),
                base_learning_rate=0.01,
                learning_rate_decay_examples=100000,
                learning_rate_decay=0.95,
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
    input_triplets: tf.placehoder. tf.float32. shape(batch_size, 3,1500)
    loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.

  """
  #create_model loss train_op add_to_collection
  result = model.create_model(input_triplets, output_size)
  
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(
    base_learning_rate,
    global_step,
    learning_rate_decay_examples,
    learning_rate_decay,
    staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)
  optimizer = optimizer_class(learning_rate)

  # for variable in slim.get_model_variables():
  #   tf.summary.histogram(variable.op.name, variable)

  output_triplets = result["output"]
  # loss = loss_fn.calculate_loss(output_triplets)
  loss, anchors, positives, negatives, pos_dist, neg_dist, hinge_dist = loss_fn.calculate_loss(output_triplets)

  reg_loss = tf.constant(0.0)
  reg_losses = tf.losses.get_regularization_losses()
  if reg_losses:
    reg_loss += tf.add_n(reg_losses)

  # Incorporate the L2 weight penalties etc.
  final_loss = regularization_penalty * reg_loss + loss
  gradients = optimizer.compute_gradients(final_loss,
              colocate_gradients_with_ops=False)
 
  tf.summary.scalar("loss", loss)
  if regularization_penalty != 0:
    tf.summary.scalar("reg_loss", reg_loss)
  
  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
      gradients = clip_gradient_norms(gradients, clip_gradient_norm)

  train_op = optimizer.apply_gradients(gradients, global_step=global_step)

  # tf.add_to_collection("input_batch", input_triplets)
  tf.add_to_collection("output_batch", output_triplets)  
  tf.add_to_collection("loss", loss)
  tf.add_to_collection("train_op", train_op)
  
  #debug
  tf.add_to_collection("gradients", gradients) 
  tf.add_to_collection("anchors",anchors)
  tf.add_to_collection("positives",positives)
  tf.add_to_collection("negatives",negatives)
  tf.add_to_collection("pos_dist",pos_dist)
  tf.add_to_collection("neg_dist",neg_dist)
  tf.add_to_collection("hinge_dist",hinge_dist)


class Trainer():

  def __init__(self, pipe, num_epochs, batch_size, wait_times, model, loss_fn, 
               checkpoint_dir, optimizer_class, config,
               last_step=None, debug=False):
    # self.is_master = (task.type == "master" and task.index == 0)
    # self.is_master = True 
    self.pipe = pipe
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.wait_times = wait_times
    self.model = model
    self.loss_fn = loss_fn
    self.checkpoint_dir = os.path.join(checkpoint_dir, model.__class__.__name__)
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

  def build_model(self,input_triplets):
    """Find the model and build the graph."""
    build_graph(input_triplets=input_triplets,
                model=self.model,
                output_size=256,
                loss_fn=self.loss_fn,
                base_learning_rate=1.0,
                learning_rate_decay_examples=10000000,
                learning_rate_decay=0.95,
                optimizer_class=self.optimizer_class,
                clip_gradient_norm=0,
                regularization_penalty=0)

  def run(self):

    self.pipe.create_pipe(self.num_epochs)

    # with tf.device('/cpu:0'):
    logging.info("Building model graph.")
    input_triplets = tf.placeholder(tf.float32, shape=(self.batch_size,3,1500), name="input_triplets")
    self.build_model(input_triplets)


    global_step = tf.train.get_or_create_global_step()
    loss = tf.get_collection("loss")[0]
    output_batch = tf.get_collection("output_batch")[0]
    train_op = tf.get_collection("train_op")[0]
    gradients = tf.get_collection("gradients")[0]
    init_op = tf.global_variables_initializer()

    anchors = tf.get_collection("anchors")[0]
    positives = tf.get_collection("positives")[0]
    negatives = tf.get_collection("negatives")[0]
    pos_dist = tf.get_collection("pos_dist")[0]
    neg_dist = tf.get_collection("neg_dist")[0]
    hinge_dist = tf.get_collection("hinge_dist")[0]

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    logging.info("Starting session.")
    with tf.Session(config=self.config) as sess:
      sess.run(init_op)
      train_writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)
      while True:
        try:
          input_triplets_val = self.pipe.get_batch(self.batch_size, self.wait_times)
          if input_triplets_val is None:
            # summary save model
            train_writer.add_summary(summary_val, global_step_val)
            saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_val)
            logging.info('Done training. Pipe end! Add summary. Save checkpoint.')
            break
          if not input_triplets_val.shape == (self.batch_size,3,1500):
            continue
          # print('input_triplets_val[0]: ',input_triplets_val[0])
          if self.debug:
            logging.debug(type(input_triplets_val)+input_triplets_val.shape+input_triplets_val.dtype)
          
          batch_start_time = time.time()
          _, global_step_val, loss_val, summary_val= sess.run(
              [train_op, global_step, loss,summary_op],
              feed_dict={input_triplets: input_triplets_val})
          # _, global_step_val, loss_val, summary_val, anchors_val, positives_val, negatives_val,pos_dist_val,neg_dist_val,hinge_dist_val = sess.run(
          #     [train_op, global_step, loss,summary_op,anchors, positives, negatives,pos_dist,neg_dist,hinge_dist],
          #     feed_dict={input_triplets: input_triplets_val})
          seconds_per_batch = time.time() - batch_start_time
            
          # print('gradients_val: ',gradients_val)
          # print('anchors_val',anchors_val)
          # print('positives_val',positives_val)
          # print('negatives_val',negatives_val)
          # print('pos_dist_val',pos_dist_val)
          # print('neg_dist_val',neg_dist_val)
          # print('hinge_dist_val',hinge_dist_val)

          logging.debug("training step " + str(global_step_val) + " | Loss: " +
              ("%.2f" % loss_val) + "\tsec/batch: " + ("%.2f" % seconds_per_batch) )

          if global_step_val % 200 == 0:
            train_writer.add_summary(summary_val, global_step_val)
            logging.info("add summary")
          if global_step_val % 1000 == 0:
            saver.save(sess, self.checkpoint_dir+'/model.ckpt', global_step_val)
            logging.info("save checkpoint")
          if global_step_val % 100000 == 0:
            # evaluate
            # _, global_step_val, loss_val, output_val = sess.run(
            #  [train_op, global_step, loss, output_batch])
            # 计算 output_batch cowatch 余弦距离
            pass

        except Exception as e:
          logging.error(str(e)) 
      
      logging.info("Exited training loop.")


def main(unused_argv):
  # TODO Prepare distributed arguments here. 
  logging.info("Tensorflow version: %s.",tf.__version__)
  checkpoint_dir = "/home/wengjy1/Checkpoints/"
  pipe = inputs.MPTripletPipe(triplet_file_patten='/data/wengjy1/cdml/*.triplet',
                              feature_file="/data/wengjy1/cdml/features.txt",
                              debug=False)
  model = find_class_by_name("VENet", [models])()
  loss_fn = find_class_by_name("HingeLoss", [losses])()
  optimizer_class = find_class_by_name("AdamOptimizer", [tf.train])
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
  config.gpu_options.allow_growth=True
  trainer = Trainer(pipe=pipe,
                    num_epochs=10,
                    batch_size=1000,
                    wait_times=100,
                    model=model,
                    loss_fn=loss_fn,
                    checkpoint_dir=checkpoint_dir,
                    optimizer_class=optimizer_class,
                    config=config,
                    last_step=None,
                    debug=False)
  trainer.run()


if __name__ == "__main__":
  tf.app.run()
