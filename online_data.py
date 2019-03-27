"""Parses the online data (stroed locally) into trainable data.
Features are dict. Guids are list.
"""
import sys
import os
import numpy as np
import json
from tensorflow import logging

from utils import exe_time
from parse_data import filter_features
from parse_data import filter_watched_guids
from parse_data import encode_base_features
from parse_data import encode_features
from parse_data import encode_all_watched_guids
from parse_data import get_all_cowatch
from parse_data import mine_triplets

logging.set_verbosity(logging.DEBUG)


def read_features_txt(filename, parse=False):
  """读取 feature txt 文件，解析并返回 dict。
  文件内容参考 tests/features.txt。
  对于每行样本，分号前是 guid, 分号后是 visual feature。
  visual feature 是 str 类型的 1500 维向量，需格式化后使用。
  Arg:
    filename: string
    parse: boolean. if set, guid will be converted to integer,
           feature will be converted to list of floats. if 
           not, guid and feature are both string.
  """
  with open(filename,'r') as file:
    features = {}
    data = file.readlines()
    for line in data:
      line = line.strip('\n')   #删除行末的 \n
      try:
        guid, feature = line.split(';')
        if parse:
          try:
            guid = int(guid)
            feature = list(map(float, (feature.split(','))))
          except Exception as e:
            logging.error('read_features_txt: '+str(e))
        features[guid]=feature
      except Exception as e:
        logging.warning(str(e)+". guid: "+guid)
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

  # filter all_watched_guids and features
  features, no_feature_guids = exe_time(filter_features)(features, all_watched_guids)
  logging.info("features size:"+str(sys.getsizeof(features))
    +"\tnumber:"+str(len(features)))
  logging.info("no_feature_guids size:"+str(sys.getsizeof(no_feature_guids))
    +"\tnumber:"+str(len(no_feature_guids)))

  all_watched_guids = exe_time(filter_watched_guids)(all_watched_guids, no_feature_guids)
  logging.info("all_watched_guids size:"+str(sys.getsizeof(all_watched_guids))
    +"\tnumber:"+str(len(all_watched_guids)))


  # encode guid to save memory and speed up processing
  encode_map, decode_map = exe_time(encode_base_features)(features)
  features = exe_time(encode_features)(features, encode_map)
  logging.info("features size:"+str(sys.getsizeof(features))
    +"\tnumber:"+str(len(features)))
  all_watched_guids = exe_time(encode_all_watched_guids)(all_watched_guids, encode_map)
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
  features_path = os.path.join(save_dir,'features.txt')
  encode_map_path = os.path.join(save_dir,'encode_map.json')
  decode_map_path = os.path.join(save_dir,'decode_map.json')
  with open(triplets_path, 'w') as file:
    for triplet in triplets:
      triplet = ','.join(list(map(str,triplet)))
      file.write(triplet+'\n')
  with open(features_path, 'w') as file:
    while features:
      guid, feature = features.popitem()
      file.write(str(guid)+';'+feature+'\n') 
  with open(encode_map_path, 'w') as file:
    json.dump(encode_map,file, ensure_ascii=False)
  with open(decode_map_path, 'w') as file:
    json.dump(decode_map, file, ensure_ascii=False)


def gen_trining_data(watch_file, feature_file, save_dir=''):
  triplets, features, encode_map, decode_map = get_triplets(watch_file, feature_file)
  write_triplets(triplets, features, encode_map, decode_map, save_dir)


if __name__ == "__main__":
  gen_trining_data(watch_file="/data/wengjy1/cdml/watched_video_ids",
                  feature_file="/data/wengjy1/cdml/video_guid_inception_feature.txt",
                  save_dir='/data/wengjy1/cdml')

