# -*- coding:utf-8 -*-
"""Provides input pipe, which can get input data tensors for models."""
import os
import time
from multiprocessing import Pool, Manager
import numpy as np
import tensorflow as tf
from tensorflow import logging
from online_data import read_features_txt

logging.set_verbosity(logging.DEBUG)
FEATURES={}

class BasePipe(object):
  """Inherit from this class when implementing new readers."""

  def create_pipe(self, unused_data, **unused_params):
    """Create the section of the graph which reads the training/val/testing data."""
    raise NotImplementedError()


class TripletPipe(BasePipe):
  def __init__(self, triplets):
    """ 该类没有 guid → feature 的映射
    Args:
      triplets: ndarray of triplets(anchor, positive, negative).
    """
    self.triplets = triplets

  def create_pipe(self, batch_size=10, num_epochs=None, num_readers=1, buffer_size=1000):
    """Construct a memory data pipe.
    Args:
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      TODO num_readers: How many I/O threads to use.
    return:
      An Iterator over the elements of this dataset. use Iterator.get_next() to
      get the next batches of triplet tensor(anchor, positive, negative)
    rasie:
    """
    dataset = tf.data.Dataset.from_tensor_slices(self.triplets)
    # Transformation: batch,shuffle,repeat
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size)
    # iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator


class MPTripletPipe(object):
  def __init__(self, triplet_file_patten, feature_file=None, features=None, debug=False):
    """
    Arg:
      triplet_file_patten: filename patten
      feature_file: filename
    """
    global FEATURES
    if feature_file is not None:
      FEATURES = read_features_txt(feature_file, parse=True)
    elif features is not None:
      FEATURES = features
    else:
      logging.error('__init__ missing 1 required positional argument, \
        at least one of \'feature_file\' and \'features\' needs to be set')
    self.triplet_files = tf.gfile.Glob(triplet_file_patten)
    logging.info(self.triplet_files)
    self.debug = debug
    if self.debug:
      logging.debug('__init__ features id: '+str(id(FEATURES)))

  def create_pipe(self, num_epochs, queue_length=2 ** 14):
    """多进程读取多个 guid_triplets 文件，在子进程中将 guid 映射成 feature
    """
    manager = Manager()
    self.triplet_queue = manager.Queue(maxsize=queue_length)
    self.mq = manager.Queue(maxsize=10)
    # self.features = manager.dict(FEATURES)
    self.pool = Pool(len(self.triplet_files))
    for index, triplet_file in enumerate(self.triplet_files):
      self.pool.apply_async(self.subprocess, args=(triplet_file, str(index),
                                            self.triplet_queue, self.mq, num_epochs,
                                            self.debug))

  @staticmethod
  def subprocess(triplet_file, thread_index, triplet_queue, mq, num_epochs, debug=False):
    """子进程为静态函数。不能用类变量，所以需要传入所需变量。"""
    global FEATURES
    if debug:
      logging.debug('thread_index: '+str(thread_index)+'; subprocess features id: '+str(id(FEATURES)))
    with open(triplet_file, 'r') as file:
      runtimes = 0
      while mq.qsize() <= 0:
        try:
          if not triplet_queue.full():
            line = file.readline()
            triplet = []
            if not line:
              if debug:
                logging.debug('thread_index: '+str(thread_index)+'; runtimes: '+str(runtimes))
              runtimes += 1
              if runtimes < num_epochs:
                file.seek(0)
                line = file.readline()
              else:
                logging.info('thread_index: '+str(thread_index)+' subprocess end')
                return
            # list of guids, dtype int
            guid_triplet = list(map(int, line.strip().split(',')))
            # if debug:
            #   logging.debug('thread_index: '+str(thread_index)+'; guid_triplet: '+str(guid_triplet))
            try:
              triplet = list(map(lambda x: FEATURES[x], guid_triplet))
            except Exception as e:
              logging.warning('subprocess: '+str(e))
              continue
            # check feature shape
            if len(triplet)==3 and len(triplet[0])==1500 and len(triplet[1])==1500 and len(triplet[2])==1500:
              triplet_queue.put(triplet)
            position = file.tell()
          else:
            time.sleep(0.01)
        except Exception as e:
          logging.warning(str(e))
          try:
            file.close()
          except Exception as e:
            pass
          try:
            file = open(file_name, 'r')
            file.seek(position)
          except Exception as e:
            logging.warning(position, str(e))
  
  def get_batch(self,batch_size, wait_times=100):
    '''get batch training data with format [arc, pos, neg]
    Arg:
      batch_size
    Retrun:
      3-D array of training triplets, dtype np.float32
    '''
    triplets = []
    wait_num = 0
    exitFlag = False
    while not exitFlag:
      if not self.triplet_queue.empty():
        wait_num = 0
        triplet = self.triplet_queue.get()
        triplets.append(triplet)
        if len(triplets) == batch_size:
          exitFlag = True
      else:
        wait_num += 1
        logging.info("queue is empty, wait:{}".format(wait_num))
        if wait_num >= wait_times:
          logging.info('queue is empty, i do not wanna to wait any more!!!')
          exitFlag = True
        time.sleep(1)
    if wait_num >= wait_times:
      return None
    return np.array(triplets, dtype=np.float32)

  def __del__(self):
    self.pool.close()
    self.pool.join()


if __name__ == '__main__':
  # test
  pipe = MPTripletPipe(triplet_file_patten='/data/wengjy1/cdml/*.triplet',
                       feature_file="/data/wengjy1/cdml/features.txt",
                       debug=True)
  pipe.create_pipe(num_epochs=2)
  # 单例
  triplet = pipe.get_batch(batch_size=50, wait_times=10)
  print(triplet.shape)
  print(triplet[0])
  # 循环
  while True:
    triplet = pipe.get_batch(batch_size=50)
    if triplet is None:
        # summary save model
        logging.info('input main: Loop end!')
        break
    if not triplet.shape[1:]==(3,1500):
      print("input main: triplet shape ERROR")