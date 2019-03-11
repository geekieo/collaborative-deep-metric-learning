# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

import losses
import inputs
import models
import utils
from utils import find_class_by_name

tf.logging.set_verbosity(logging.DEBUG)
FLAGS = tf.flags.FLAGS

flags.DEFINE_string(
    "train_dir", "/model/imitation/",
    "The directory to save the model files in.")
flags.DEFINE_string(
    "pipe", "TripletPipe",
    "The iterator which reads the training data"


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
                model,
                data,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
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
    pipe: The data file reader. It should inherit from BasePipe.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    data: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
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
  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  gpus = gpus[:FLAGS.num_gpu]
  num_gpus = len(gpus)
  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:%d'
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:%d'

  learning_rate = tf.train.exponential_decay(
    base_learning_rate,
    global_step * batch_size * num_towers,
    learning_rate_decay_examples,
    learning_rate_decay,
    staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)
  optimizer = optimizer_class(learning_rate)
  triplets = pipe.create_pipe(features, batch_size=batch_size, num_epochs=None)

  #create_model loss train_op add_to_collection
  





class Trainer():

  def __init__(self, train_dir, model, pipe)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement)
    self.model = model
    self.tringinpipe = pipe

  
  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""
    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def build_model(self, model, pipe):
    """Find the model and build the graph."""
    label_loss_fn = find_class_by_name(FLAGS.loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])
    build_graph(reader=reader,
                model=model,
                optimizer_class=optimizer_class,
                clip_gradient_norm=FLAGS.clip_gradient_norm,
                train_data_pattern=FLAGS.train_data_pattern,
                label_loss_fn=label_loss_fn,
                base_learning_rate=FLAGS.base_learning_rate,
                learning_rate_decay=FLAGS.learning_rate_decay,
                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                regularization_penalty=FLAGS.regularization_penalty,
                num_readers=FLAGS.num_readers,
                batch_size=FLAGS.batch_size,
                num_epochs=FLAGS.num_epochs)
    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)
    
  def run(self):
    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)
    target, device_fn = self.start_server_if_distributed()
    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("output")[0]
        train_op = tf.get_collection("train_op")[0]
        init_op = tf.global_variables_initializer()

    supervisor = tf.train.Supervisor(
      graph,
      logdir=self.train_dir,
      init_op=init_op,
      is_chief=self.is_master,
      global_step=global_step,
      save_model_secs=15 * 60,
      save_summaries_secs=120,
      saver=saver)


def main():
  logging.info("Tensorflow version: %s.",tf.__version__)
  model = find_class_by_name(FLAGS.model, [models])()
  pipe = inputs.TripletPipe()
  trainer = Trainer(FLAGS.train_dir, model, pipe,
          FLAGS.log_device_placement, FLAGS.max_steps,
          FLAGS.export_model_steps)
  trainer.run()  

if __name__ == "__main__":
  tf.app.run()
