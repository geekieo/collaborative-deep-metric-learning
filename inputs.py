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
    def __init__(self, triplet_file_patten, feature_file, debug=False):
        """
        Arg:
          triplet_file_patten: filename patten
          feature_file: filename
        """
        self.triplet_files = tf.gfile.Glob(triplet_file_patten)
        self.features = read_features_txt(feature_file, parse=True)
        self.debug = debug
        if debug:
            logging.debug(self.triplet_files)
            logging.debug('__init__'+str(id(self.features)))
        
    def run_multiporcess(self, num_epochs, queue_length = 2 ** 14):
        manager = Manager()
        self.triplet_queue = manager.Queue(maxsize=queue_length)
        self.mq = manager.Queue(maxsize=10)
        self.pool = Pool(len(self.triplet_files))
        for index,triplet_file in enumerate(self.triplet_files):
            self.pool.apply_async(self.process, args=(triplet_file, self.features, str(index),
                                                      self.triplet_queue, self.mq, num_epochs,
                                                      self.debug))
    

    @staticmethod
    def process(triplet_file, features, thread_index, triplet_queue, mq, num_epochs=1,debug=False):
        """
        子进程为静态函数。
        共享变量。
        不能用类变量，所以需要传入所有变量
        """
        if debug:
            logging.debug('process features id:'+str(id(features)))
        with open(triplet_file, 'r') as file:
            logging.info(thread_index)
            runtimes = 0
            while mq.qsize() <= 0:
                try:
                    if not triplet_queue.full():
                        line = file.readline()
                        if not line:
                            runtimes += 1
                            if runtimes < num_epochs:
                                file.seek(0)
                                line = file.readline()
                            else:
                                return
                        triplet = list(map(int,line.strip().split(','))) # list of guids, dtype int
                        if debug:
                            logging.debug(str(triplet))
                        triplet = list(map(lambda x: features[x], triplet))
                        if triplet is None:
                            continue
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

    def create_pipe(self, batch_size=50, num_epochs=2):
        '''get batch training data with format [arc, pos, neg]
        Arg:
          batch_size
        Retrun:
          a batch of training triplets, 
        '''
        self.run_multiporcess(num_epochs)
        triplets=[]
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
                if wait_num >= 100:
                    print('queue is empty, i do not wanna to wait any more!!!')
                    exitFlag = True
                # queueLock.release()
                print("queue is empty, wait:{}".format(wait_num))
                time.sleep(1)
        if wait_num >= 100:
            return None
        return np.array(triplets)
    
    def __del__(self):
        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    # process/pool shared python object!!!
    pipe = MPTripletPipe(triplet_file_patten='tests/*.triplet',
                         feature_file="tests/features.txt",
                         debug=True)

    triplet = pipe.create_pipe(batch_size=5, num_epochs=2)
    if triplet is not None:
        print(triplet.shape)
        print(triplet[0])
