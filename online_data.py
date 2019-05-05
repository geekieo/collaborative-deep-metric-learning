# -*- coding:utf-8 -*-
"""Parses the online data (stroed locally) into trainable data.
Features are dict. Guids are list.
"""
import sys
import os
import numpy as np
import json
from tensorflow import logging
import collections

from utils import exe_time
from parse_data import get_unique_watched_guids
from parse_data import filter_features
from parse_data import filter_watched_guids
from parse_data import encode_base_features
from parse_data import encode_features
from parse_data import encode_all_watched_guids
from parse_data import get_all_cowatch
from parse_data import get_cowatch_graph
from parse_data import select_cowatch
from parse_data import mine_triplets

logging.set_verbosity(logging.DEBUG)


def read_features_txt(filename):
  """读取 feature txt 文件，解析并返回 dict。
  文件内容参考 tests/features.txt。
  对于每行样本，分号前是 guid, 分号后是 visual feature。
  visual feature 是 str 类型的 1500 维向量，需格式化后使用。
  Arg:
    filename: string
  """
  with open(filename,'r') as file:
    features = collections.OrderedDict()
    data = file.readlines()
    for line in data:
      line = line.strip('\n')   #删除行末的 \n
      try:
        str_guid, str_feature = line.split(';')
        try:
          feature = list(map(float, (str_feature.split(','))))
          if len(feature) == 1500:
            features[str_guid]=feature
        except Exception as e:
          logging.warning('read_features_txt: drop feature. '+str(e))
      except Exception as e:
        logging.warning('read_features_txt'+str(e))
    return features


def read_features_npy(filename):
  """features 必须是 key 从 0 自增的 dict，其 value 为 ndarray。这里
  把 features 按 key 的顺序存成 ndarray，其索引天然为key。用 ndarray
  的索引批量查询 ndarray 能提高 batch 查询速度。
  """
  features = np.load(filename)
  return features
  

# def read_features_json(filename):
#   with open(filename, 'r') as file:
#     features = json.load(file)
#     return features


def read_watched_guids(filename):
  """读取 watched_guids txt 文件, 返回 list
  文件内容参考 tests/watched_guids.txt
  对于每行样本，以逗号分隔每个元素，第一个元素为用户 uid，
  之后的每个元素为按顺序观看的 guid。丢弃 uid, 仅取 guids 
  作为待返回 list 中的一个元素。
  retrun:
    all_watched_guids: 2-D list contains wathced guids of each user
  """
  with open(filename, 'r') as file:
    all_watched_guids=[]
    data = file.readlines()
    for line in data:
      line = line.rstrip('\n')  # 删除行末的 \n
      guids = line.split(',')
      try:
        guids = guids[1:]       # 删除第一个元素 uid
        watched_guids = []
        for i,guid in enumerate(guids):
          if i>0 and guids[i]==guids[i-1]:
            continue            # 过滤相邻重复元素
          if guid[:6]=='video_':
            guid = guid[6:]     # 删除每个 guid 前缀"video_"
          else:
            logging.warning('存在没有前缀 "video_" 的 guid in watched_guids: '+guid)
          watched_guids.append(guid)
        if len(watched_guids)>1:
          all_watched_guids.append(watched_guids)
      except Exception as e:
        logging.warning(str(e)+". uid: "+guids[0])
    return all_watched_guids


def load_cowatches(filename):
  '''
  Return: list of cowatch index pair.
  '''
  cowatches = []
  with open(filename, 'r') as file:
    for line in file.readlines():
      cowatch = []
      line = line.strip()
      ids = line.split(',')
      try:
        cowatch.append(int(ids[0]))
        cowatch.append(int(ids[1]))
        cowatches.append(cowatch)
      except Exception as e:
        logging.warning(str(e))
  logging.info('online_data load_cowatches len:'+str(len(cowatches)))
  return cowatches
  

def get_cowatches(watch_file, feature_file):
  """
  Args:
    watch_file: str. file path of watched guid file.
    feature_file: str. file path of video feature.
    threshold:int. a threshold of cowatch number use to select cowatch pair
  Return:
    cowatches: list of guid cowatches
    features: dict of features
  """
  # read file
  logging.info('read features txt...')
  features = exe_time(read_features_txt)(feature_file)  #345.779s
  logging.info("features size:"+str(sys.getsizeof(features))+"\tnum:"+str(len(features)))

  all_watched_guids = exe_time(read_watched_guids)(watch_file) #215.065s
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))+"\tnum:"+str(len(all_watched_guids)))

  # filter all_watched_guids and features
  features, no_feature_guids = exe_time(filter_features)(features, all_watched_guids) #41.599s
  logging.info("features size:"+str(sys.getsizeof(features))+"\tnum:"+str(len(features)))
  logging.info("no_feature_guids size:"+str(sys.getsizeof(no_feature_guids))+"\tnum:"+str(len(no_feature_guids)))

  all_watched_guids = exe_time(filter_watched_guids)(all_watched_guids, no_feature_guids) #94.864s
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))+"\tnum:"+str(len(all_watched_guids)))

  # encode guid to sequential integer, to save memory and speed up processing
  encode_map, decode_map = exe_time(encode_base_features)(features) #1.134s
  features = exe_time(encode_features)(features, decode_map)  #422.296s #TODO
  logging.info("features size:"+str(sys.getsizeof(features))+"\tnum:"+str(len(features)))
  
  all_watched_guids = exe_time(encode_all_watched_guids)(all_watched_guids, encode_map) #146.209s
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))+"\tnum:"+str(len(all_watched_guids)))

  # get cowatch
  all_cowatch = exe_time(get_all_cowatch)(all_watched_guids)  #283.462s
  logging.info("all_cowatch size:"+str(sys.getsizeof(all_cowatch))+"\tnum:"+str(len(all_cowatch)))
  
  # get cowatch graph, delete self pair
  graph, cowatches = exe_time(get_cowatch_graph)(all_cowatch) #297.414s

  return cowatches, features, encode_map, decode_map, graph


def select_cowatches(cowatches, graph,threshold=3, unique=False):
  cowatches = select_cowatch(graph, threshold, cowatches, unique=unique)
  logging.info("cowatches size:"+str(sys.getsizeof(cowatches))+"\tnum:"+str(len(cowatches)))
  logging.debug("unique_guids in cowatches:"+str(len(exe_time(get_unique_watched_guids)(cowatches))))
  return cowatches

@DeprecationWarning
def get_triplets(watch_file, feature_file, threshold=3,unique=False):
  """
  Args:
    watch_file: str. file path of watched guid file.
    feature_file: str. file path of video feature.
    threshold:int. a threshold of cowatch number use to select cowatch pair
  Return:
    return: list of guid_triplets
    features: dict of features
  """
  cowatches, features, encode_map, decode_map, graph = get_cowatches(watch_file, feature_file)
  cowatches = select_cowatches(cowatches, graph, threshold, unique=unique)
  # mine triplets
  triplets = exe_time(mine_triplets)(cowatches, features)
  logging.info("triplets size:"+str(sys.getsizeof(triplets))+"\tnum:"+str(len(triplets)))
  
  return triplets, features, encode_map, decode_map


def write_features(features, encode_map=None, decode_map=None, save_dir=''):
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  features_path = os.path.join(save_dir,'features.npy')
  encode_map_path = os.path.join(save_dir,'encode_map.json')
  decode_map_path = os.path.join(save_dir,'decode_map.json')

  try:
    np.save(features_path, features)
    logging.info("Write features success!")
    if encode_map is not None:
      with open(encode_map_path, 'w') as file:
        json.dump(encode_map,file, ensure_ascii=False)
        logging.info("Write encode_map success!")
    if encode_map is not None:
      with open(decode_map_path, 'w') as file:
        json.dump(decode_map, file, ensure_ascii=False)
      logging.info("Write decode_map success!")
    return True
  except Exception as e:
    logging.warning(str(e))
    return False


@DeprecationWarning
def write_triplets(triplets, save_dir='',split=4, ):
  """
  args:
    triplets: list of str
  """
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  triplets_path = os.path.join(save_dir,'triplets.txt')
  with open(triplets_path, 'w') as file:
    for triplet in triplets:
      triplet = ','.join(list(map(str,triplet)))
      file.write(triplet+'\n')
  try:
    split = 1 if split<1 else int(split)  
    row_num = len(triplets)
    row_cnt = int(row_num / split) if row_num % split == 0 else int(row_num / split)+1
    cwd = os.getcwd()
    os.chdir(save_dir)
    command = "split -l %d triplets.txt --additional-suffix=.triplet" % (row_cnt) 
    os.system(command)
    os.chdir(cwd)
    return True
  except Exception as e:
    logging.warning(str(e))
    return False

def write_cowatches(cowatches, save_dir='',split=4, eval_num=100000, test_num=100000):
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  eval_data_path = os.path.join(save_dir,'cowatches.eval')
  test_data_path = os.path.join(save_dir,'cowatches.test')
  train_data_path = os.path.join(save_dir,'cowatches.txt')
  if eval_num + test_num > 0.3*len(cowatches):
    eval_num = int(len(cowatches)*0.15)
    test_num = int(len(cowatches)*0.15)
  try:
    # split cowatch data to training data, evaluation data, testing data
    if eval_num:
      with open(eval_data_path, 'w') as file:
        for cowatch in cowatches[:eval_num]:
          cowatch = ','.join(list(map(str,cowatch)))
          file.write(cowatch+'\n')
    if test_num:
      with open(test_data_path, 'w') as file:
        for cowatch in cowatches[eval_num:eval_num+test_num]:
          cowatch = ','.join(list(map(str,cowatch)))
          file.write(cowatch+'\n')
    with open(train_data_path, 'w') as file:
      for cowatch in cowatches[eval_num+test_num:]:
        cowatch = ','.join(list(map(str,cowatch)))
        file.write(cowatch+'\n')
    # split training data to multi part
    split = 1 if split<1 else int(split)  
    row_num = len(cowatches)
    row_cnt = int(row_num / split) if row_num % split == 0 else int(row_num / split)+1
    cwd = os.getcwd()
    os.chdir(save_dir)
    command = "split -l %d cowatches.txt --additional-suffix=.train" % (row_cnt)
    os.system(command)
    os.system("rm -f cowatches.txt")
    os.chdir(cwd)
    logging.info("Write cowatch data success")
    return True
  except Exception as e:
    logging.error(str(e)+' write cowatches failed.')
    return False


def gen_training_data(watch_file, feature_file,threshold=3, base_save_dir='/.',split=4, unique=False):
  all_cowatch, features, encode_map, decode_map, graph = get_cowatches(watch_file, feature_file)
  thresholds = threshold+1
  for threshold in range(1,thresholds):
    for unique in [True, False]:
      save_dir = os.path.join(base_save_dir,'cdml_'+str(threshold)+('_unique' if unique else ''))
      if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        if not os.path.exists(save_dir):
          logging.error('Can not make dir:'+str(save_dir))
          return
      res1 = exe_time(write_features)(features, encode_map, decode_map, save_dir) #7.333s
      cowatches = exe_time(select_cowatches)(all_cowatch, graph, threshold, unique) #44.866s
      res2 = exe_time(write_cowatches)(cowatches, save_dir,split) #15.130s
      if res1 and res2:
        logging.info("Training data have saved to: "+save_dir)


if __name__ == "__main__":
  gen_training_data(watch_file="/data/wengjy1/cdml/watched_video_ids",
                    feature_file="/data/wengjy1/cdml/video_guid_inception_feature.txt",
                    threshold = 3,
                    base_save_dir='/data/wengjy1/',
                    split=10,
                    unique=False)
  
