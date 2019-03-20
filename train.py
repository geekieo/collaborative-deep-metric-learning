# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import logging
# from tensorflow import flags
# from tensorflow.python.client import device_lib

import losses
import inputs
import models
import utils
from utils import find_class_by_name

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


def build_graph(pipe,
                data,
                model,
                output_size=256,
                loss_fn=losses.HingeLoss(),
                batch_size=100,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_epochs=None,
                num_readers=1,):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    pipe: The data pipe. It should inherit from BasePipe.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    data: glob path to the training data files.
    loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """
  global_step = tf.Variable(0, trainable=False, name="global_step")

  learning_rate = tf.train.exponential_decay(
    base_learning_rate,
    global_step * batch_size,
    learning_rate_decay_examples,
    learning_rate_decay,
    staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)
  input_iter = pipe.create_pipe(data, batch_size=batch_size, num_epochs=num_epochs)
  input_triplets = input_iter.get_next()
  input_triplets = tf.cast(input_triplets, tf.float32)
  # tf.summary.histogram("input_triplets", input_triplets)

  #create_model loss train_op add_to_collection
  result = model.create_model(input_triplets, output_size)
  for variable in slim.get_model_variables():
    tf.summary.histogram(variable.op.name, variable)
  output_triplets = result["output"]
  loss = loss_fn.calculate_loss(output_triplets)

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

  # optimizer 会为 global_step 做自增操作
  train_op = optimizer.apply_gradients(gradients, global_step=global_step)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("input_batch", input_triplets)
  tf.add_to_collection("output_batch", output_triplets)  
  tf.add_to_collection("loss", loss)
  tf.add_to_collection("train_op", train_op)


class Trainer():

  def __init__(self, checkpoint_dir, data, pipe, model, loss_fn, optimizer_class,
               batch_size, num_epochs=None, log_device_placement=True, last_step=None):
    # self.is_master = (task.type == "master" and task.index == 0)
    self.is_master = True 
    self.checkpoint_dir = checkpoint_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement)
    self.model = model
    self.pipe = pipe
    self.data = data
    self.loss_fn = loss_fn
    self.optimizer_class = optimizer_class
    self.batch_size = batch_size
    self.num_epochs = num_epochs

    self.last_step = last_step
    
  def build_model(self):
    """Find the model and build the graph."""
    build_graph(data=self.data,
                pipe=self.pipe,
                model=self.model,
                output_size=256,
                loss_fn=self.loss_fn,
                batch_size=self.batch_size,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=self.optimizer_class,
                clip_gradient_norm=0,
                regularization_penalty=0,
                num_epochs=self.num_epochs,
                num_readers=1)

  def run(self):
    with tf.Graph().as_default() as graph:
      self.build_model()

      global_step = tf.get_collection("global_step")[0]
      loss = tf.get_collection("loss")[0]
      output_batch = tf.get_collection("output_batch")[0]
      train_op = tf.get_collection("train_op")[0]
      init_op = tf.global_variables_initializer()

      hooks = [tf.train.NanTensorHook(loss)]
      if self.last_step:
        hooks.append(tf.train.StopAtStepHook(last_step=self.last_step))

      logging.info("%s: Starting monitored session.")
      with tf.train.MonitoredTrainingSession(checkpoint_dir = self.checkpoint_dir,
                                             hooks = hooks,
                                             save_summaries_steps=100,
                                             save_checkpoint_secs=600) as sess:

        while not sess.should_stop():
          batch_start_time = time.time()
          _, global_step_val, loss_val,output_val = sess.run(
              [train_op, global_step, loss, output_batch])
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = output_val.shape[0] / seconds_per_batch

          if global_step_val % 10 == 0 and self.checkpoint_dir:
            logging.info("training step " + str(global_step_val) + " | Loss: " +
              ("%.2f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))
            # model.evaluate()
          else:
            logging.info("training step " + str(global_step_val) + " | Loss: " +
              ("%.2f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))

      logging.info("Exited training loop.")




def main(unused_argv):
  # TODO Prepare distributed arguments here. 

  # from imitation_data import gen_triplets
  # data = gen_triplets(batch_size=3000,feature_size=1500)

  from online_data import get_triplets
  triplets = get_triplets(watch_file="/data/wengjy1/watched_video_ids",
                          feature_file="/data/wengjy1/video_guid_inception_feature.txt",
                          return_features=True)
  triplets=np.array(triplets, dtype=np.float32)
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
                    num_epochs=1
                    )
  trainer.run() 

if __name__ == "__main__":
  tf.app.run()
