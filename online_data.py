"""Parses the online data (stroed locally) into trainable data.
Features are dict. Guids are list.
"""
import sys
import json
from tensorflow import logging

from utils import exe_time
from parse_data import get_all_cowatch
from parse_data import mine_triplets

logging.set_verbosity(logging.DEBUG)

def read_features_txt(filename):
  """读取 feature txt 文件，解析并返回 dict。
  文件内容参考 tests/features.txt。
  对于每行样本，分号前是 guid, 分号后是 visual feature，
  visual feature 是 1500 维的浮点向量。
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
        logging.warning(e+". Context: "+line)
  return features


def trans_features_to_json(filename, save_dir):
  """将 feature txt 文件转换成 json 格式。"""
  features = read_features_txt(filename)
  with open(save_dir, 'w') as file:
    json.dump(features, file)
  

def read_features_json(filename):
  """读取 feature json 文件，返回 dict。
  文件内容参考 tests/features.json。
  对于每行样本，分号前是 guid, 分号后是 visual feature，
  visual feature 是 1500 维的浮点向量。
  """
  with open(filename, 'r') as file:
    features = json.load(file)
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
      try:
        guids = line.split(',')
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
        logging.warning(e+". Context: "+line)
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
  删除 features 中用 all_watched_guids 访问不到的 feature
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
      watched_features[guid]=features[guid]
    except KeyError as e:
      # logging.warning(str(e)+"  guid: "+str(guid))
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
    return: triplets. ndarray of [anchor feature, positive feature, negative feature]
  """
  all_watched_guids = read_watched_guids(watch_file)
  features = read_features_txt(feature_file)

  # filter all_watched_guids and features
  features, no_feature_guids = filter_features(features, all_watched_guids)
  all_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)

  # mine triplets
  all_cowatch = exe_time(get_all_cowatch)(all_watched_guids)
  logging.info("Memory:"+str(sys.getsizeof(all_cowatch))
    +"\tNum:"+str(len(all_cowatch)))
  triplets = exe_time(mine_triplets)(all_cowatch, features)
  logging.info("Memory:"+str(sys.getsizeof(triplets)))
  return triplets
  
  
