# -*- coding:utf-8 -*-
"""Provides input pipe, which can get input data tensors for models."""
import os
import time
from multiprocessing import Pool
from multiprocessing import Manager
# from multiprocessing import Value
import numpy as np
import tensorflow as tf
from tensorflow import logging
from online_data import read_features_npy
from parse_data import yield_negative_index

logging.set_verbosity(logging.DEBUG)
FEATURES={}
# ALL_FILES_NUM = 0
# FINISHED_NUM = Value("i", 0) #


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
  def __init__(self, cowatch_file_patten, feature_file=None):
    """
    Arg:
      cowatch_file_patten: filename patten
      feature_file: filename
    """
    global FEATURES
    # global ALL_FILES_NUM
    FEATURES = read_features_npy(feature_file)
    
    self.cowatch_files = tf.gfile.Glob(cowatch_file_patten)
    # ALL_FILES_NUM = len(self.cowatch_files)
    logging.info('MPTripletPipe __init__ cowatch_files: '+str(self.cowatch_files))
    logging.debug('MPTripletPipe __init__ features id: '+str(id(FEATURES)))

  def create_pipe(self, num_epochs, batch_size, queue_length=2 ** 14):
    """多进程读取多个 guid_triplets 文件，在子进程中将 guid 映射成 feature
    """
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    manager = Manager()
    self.triplet_queue = manager.Queue(maxsize=queue_length)
    self.mq = manager.Queue(maxsize=10)
    self.pool = Pool(len(self.cowatch_files))
    for index, cowatch_file in enumerate(self.cowatch_files):
      self.pool.apply_async(self.subprocess, args=(cowatch_file, str(index),
                                                   self.triplet_queue, self.mq,
                                                   self.num_epochs, self.batch_size))

  @staticmethod
  def subprocess(cowatch_file, thread_index, triplet_queue, mq, num_epochs, batch_size):
    """子进程为静态函数。不能用类变量，所以需要传入所需变量。"""    
    neg_iter = yield_negative_index(len(FEATURES), putback=True)
    with open(cowatch_file, 'r') as file:
      runtimes = 0
      triplets = []
      position = 0
      while True:
        try:
          for i in range(batch_size):
            line = file.readline()
            if not line:
              logging.debug('thread_index: '+str(thread_index)+'; runtimes: '+str(runtimes))
              runtimes += 1
              if runtimes < num_epochs:
                # global FINISHED_NUM
                # added = False
                # while True:
                #   with FINISHED_NUM.get_lock():
                #     if added == False:
                #       added = True
                #       FINISHED_NUM.value += 1
                #     if FINISHED_NUM.value == ALL_FILES_NUM * runtimes:
                #       break
                #   logging.debug('thread_index: ' + str(thread_index) + ' waiting partners...')
                #   time.sleep(2)
                file.seek(0)
                line = file.readline()
              else:
                logging.info('thread_index: '+str(thread_index)+' subprocess end')
                return
            arc, pos = line.strip().split(',')
            triplet = [int(arc), int(pos)]
            neg = neg_iter.__next__()
            while neg in triplet:
              neg = neg_iter.__next__()
            triplet.append(neg)
            triplets.append(triplet)
          triplet_queue.put(triplets)
          triplets.clear()
        except Exception as e:
          logging.warning('subprocess:'+str(e))
          try:
            file.close()
          except Exception as e:
            pass
          try:
            file = open(file_name, 'r')
            file.seek(position)
          except Exception as e:
            logging.warning('subprocess:'+str(position)+str(e))

  def get_batch(self, wait_times=30):
    '''get batch training data with format [arc, pos, neg]
    Retrun:
      3-D array of training triplets, dtype np.float32
    '''
    triplets = []
    wait_num = 0
    exitFlag = False
    while not exitFlag:
      if not self.triplet_queue.empty():
        wait_num = 0
        guid_triplets = self.triplet_queue.get()
        if len(guid_triplets) == self.batch_size:
          return FEATURES[np.asarray(guid_triplets)]
      else:
        wait_num += 1
        logging.info("queue is empty, wait:{}".format(wait_num))
        if wait_num >= wait_times:
          logging.info('queue is empty, i do not wanna to wait any more!!!')
          exitFlag = True
        time.sleep(1)
    return
    
  def __del__(self):
    self.pool.close()
    self.pool.join()


if __name__ == '__main__':
  # test
  pipe = MPTripletPipe(triplet_file_patten='/data/wengjy1/cdml/*.cowatch',
                       feature_file="/data/wengjy1/cdml/features.txt")
  pipe.create_pipe(num_epochs=2,batch_size=50)
  # 单例
  triplet = pipe.get_batch(wait_times=10)
  print(triplet.shape)
  print(triplet[0])
  # 循环
  while True:
    triplet = pipe.get_batch(wait_times=10)
    if triplet is None:
        # summary save model
        logging.info('input main: Loop end!')
        break
    if not triplet.shape[1:]==(3,1500):
      print("input main: triplet shape ERROR")