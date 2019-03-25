"""Parses the online data (stroed locally) into trainable data.
Features are dict. Guids are list.
"""
import sys
import os
import numpy as np
import json
from tensorflow import logging

from utils import exe_time
from parse_data import encode_base_features
from parse_data import encode_features
from parse_data import encode_all_watched_guids
from parse_data import get_all_cowatch
from parse_data import mine_triplets

logging.set_verbosity(logging.DEBUG)


def read_features_txt(filename):
  """读取 feature txt 文件，解析并返回 dict。
  文件内容参考 tests/features.txt。
  对于每行样本，分号前是 guid, 分号后是 visual feature。
  visual feature 是 str 类型的 1500 维向量，需格式化后使用。
  """
  features = {}
  with open(filename,'r') as file:
    data = file.readlines()
    for line in data:
      line = line.strip('\n')   #删除行末的 \n
      try:
        guid, feature = line.split(';')
        features[guid]=feature
      except Exception as e:
        logging.warning(str(e)+". guid: "+guid)
  return features


def read_watched_guids(filename):
  """读取 watched_guids txt 文件, 返回 list
  文件内容参考 tests/watched_guids.txt
  对于每行样本，以逗号分隔每个元素，第一个元素为用户 uid，
  之后的每个元素为按顺序观看的 guid。丢弃 uid, 仅取 guids 
  作为待返回 list 中的一个元素。
  retrun:
    all_watched_guids: 2-D list contains wathced guids of each user
  """
  all_watched_guids=[]
  with open(filename, 'r') as file:
    data = file.readlines()
    for line in data:
      line = line.rstrip('\n')  # 删除行末的 \n
      guids = line.split(',')
      try:
        guids = guids[1:]       # 删除第一个元素 uid
        watched_guids = []
        for guid in guids:
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


def get_unique_watched_guids(all_watched_guids):
  """get unique guids from all wathced guids.
  用于取得 watched_guids 和 features 的交集
  return:
    flatten list of unique_watched_guids
  """
  guids = [guid for watched_guids in all_watched_guids 
                  for guid in watched_guids]
  unique_watched_guids = list(set(guids))
  return unique_watched_guids


def filter_features(features, all_watched_guids):
  """Filter features by the key of unique_guids
  删除 features 中用 all_watched_guids 没访问的 feature
  并找出无法获取到 feature 的 guid 
  Args:
    features: dict
    all_watched_guids: 2-D list. list of watched guids.
  Return:
    watched_features: dict
    no_feature_guids: list
  """
  unique_watched_guids = get_unique_watched_guids(all_watched_guids)
  no_feature_guids=[]
  watched_features={}
  for guid in unique_watched_guids:
    try:
      watched_features[guid]=features.pop(guid)
    except KeyError as e:
      no_feature_guids.append(guid)
  return watched_features, no_feature_guids


def filter_watched_guids(all_watched_guids, no_feature_guids):
  """ Filter all_watched_guids by no_feature_guids.
  删除 all_watched_guids 中访问不到 feature 的 guid，
  并以此 guid 为分割点，将这一条 watched_guids 分成两条。
  用于生成可成功获取 feature 的 cowatch。
  """
  no_feature_guids = set(no_feature_guids)
  filtered_watched_guids=[]
  for watched_guids in all_watched_guids:
    # 切分出的列表的头元素在原列表的索引
    head = 0
    for i,guid in enumerate(watched_guids):
      if guid in no_feature_guids:
        # 忽略长度小于2的列表，无法生成 cowatch
        if len(watched_guids[head:i]) > 1:
          filtered_watched_guids.append(watched_guids[head:i])
        head = i+1
    # 忽略长度小于2的列表，无法生成 cowatch
    if len(watched_guids[head:]) > 1:
      filtered_watched_guids.append(watched_guids[head:])
  return filtered_watched_guids


def get_triplets(watch_file, feature_file):
  """
  Args:
    watch_file: str. file path of watched guid file.
    feature_file: str. file path of video feature.
  Return:
    return: list of guid_triplets
    features: dict of features
  """
  # read file
  features = exe_time(read_features_txt)(feature_file)
  logging.info("features size:"+str(sys.getsizeof(features))
    +"\tnumber:"+str(len(features)))

  all_watched_guids = exe_time(read_watched_guids)(watch_file)
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))
    +"\tnumber:"+str(len(all_watched_guids)))

  # # encode guid to save memory and speed processing
  encode_map, decode_map = exe_time(encode_base_features)(features)
  features = exe_time(encode_features)(features, encode_map)
  logging.info("features size:"+str(sys.getsizeof(features))
    +"\tnumber:"+str(len(features)))
  all_watched_guids = exe_time(encode_all_watched_guids)(all_watched_guids, encode_map)
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))
    +"\tnumber:"+str(len(all_watched_guids)))

  # filter all_watched_guids and features
  features, no_feature_guids = exe_time(filter_features)(features, all_watched_guids)
  logging.info("features size:"+str(sys.getsizeof(features))
    +"\tnumber:"+str(len(features)))
  logging.info("no_feature_guids size:"+str(sys.getsizeof(no_feature_guids))
    +"\tnumber:"+str(len(no_feature_guids)))

  all_watched_guids = exe_time(filter_watched_guids)(all_watched_guids, no_feature_guids)
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))
    +"\tnumber:"+str(len(all_watched_guids)))

  # select co_watch pair
  # TODO

  # mine triplets
  all_cowatch = exe_time(get_all_cowatch)(all_watched_guids)
  logging.info("all_cowatch size:"+str(sys.getsizeof(all_cowatch))
    +"\tnumber:"+str(len(all_cowatch)))
      
  triplets = exe_time(mine_triplets)(all_cowatch, features)
  logging.info("triplets size:"+str(sys.getsizeof(triplets))
    +"\tnumber:"+str(len(triplets)))
  
  return triplets, features, encode_map, decode_map
  

def write_triplets(triplets, features, encode_map, decode_map, save_dir=''):
  triplets_path = os.path.join(save_dir,'triplets.txt')
  features_path = os.path.join(save_dir,'features.json')
  encode_map_path = os.path.join(save_dir,'encode_map.json')
  decode_map_path = os.path.join(save_dir,'decode_map.json')
  with open(triplets_path, 'w') as file:
    for triplet in triplets:
      triplet = ','.join(list(map(str,triplet)))
      file.write(triplet+'\n')
  with open(features_path, 'w') as file:
    json.dump(features, file, ensure_ascii=False)
  with open(encode_map_path, 'w') as file:
    json.dump(encode_map,file, ensure_ascii=False)
  with open(decode_map_path, 'w') as file:
    json.dump(decode_map, file, ensure_ascii=False)


def gen_trining_data(watch_file, feature_file, save_dir=''):
  triplets, features, encode_map, decode_map = get_triplets(watch_file, feature_file)
  write_triplets(triplets, features, encode_map, decode_map, save_dir)


if __name__ == "__main__":
  gen_trining_data(watch_file="/data/wengjy1/watched_video_ids",
                  feature_file="/data/wengjy1/video_guid_inception_feature.txt")

