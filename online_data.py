"""Parses the online data stored locally into trainable data
Features are dict. Guids are list.
"""
from tensorflow import logging
import json
# import parse_data import get_all_cowatch

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
  """get unique guids from all wathced guids"""
  guids = [guid for watched_guids in all_watched_guids 
                  for guid in watched_guids]
  unique_guids = list(set(guids))
  return unique_guids


def get_watched_features(unique_guids, features):
  """get feature from features by the key of unique_guids"""
  watched_features={}
  for guid in unique_guids:
    try:
      watched_features[guid]=features[guid]
    except KeyError as e:
      # logging.warning(str(e)+"  guid: "+str(guid))python
      pass
  return watched_features

def get_cowatch(all_watched_guids, features):
  pass
  
