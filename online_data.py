"""Parses the online data stored locally into trainable data"""
import json


def read_features_txt(filename):
  """读取 feature txt 文件，解析并返回 dict。
  文件内容参考 tests/features.txt。
  对于每行样本，分号前是 guid, 分号后是 visual feature，
  visual feature 是 1500 维的浮点向量。
  """
  with open(filename,'r') as file:
    data = file.readlines()
    features = {}
    for line in data:
      guid, feature = line.split(';')
      features[guid]=feature
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
  """读取 watched_guids txt 文件，返回 watch_guids list
  文件内容参考 tests/watched_guids.txt
  对于每行样本，以逗号分隔每个元素，第一个元素为用户 uid，
  之后的每个元素为按顺序观看的 guid,这些 guids 组成了待返回
  list 中的一个元素。
  """
  
  return all_watched_guids
    