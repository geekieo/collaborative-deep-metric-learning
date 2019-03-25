# -*- coding:utf-8 -*-
""" Parse watched_guids to triplets with features
本模块的主要任务有：压缩编码 guids。挖掘 cowatch，统计 cowatch。随机匹配 
negative，生成压缩编码的 guid triplets，以及对应压缩键值的 features。假设 
watched_guids 可以访问到 features 中的所有feature。
"""
import numpy as np
from itertools import cycle
import copy
from tensorflow import logging
from tqdm import tqdm

logging.set_verbosity(logging.DEBUG)


# ========================= encoding guid =========================
def encode_guids(guids):
  """将 str guids 编码成有序的 index, 并返回编码/解码字典"""
  encode_map={}
  decode_map={}
  for i,guid in enumerate(guids):
    if not guid in encode_map:
      encode_map[guid]=i
      decode_map[i]=guid
  return encode_map, decode_map

def encode_base_features(features):
  """ 以 features 的 keys 为基准，编码 guids"""
  encode_map, decode_map = encode_guids(features.keys())
  return encode_map, decode_map


def encode_features(features, encode_map):
  """ 编码 features 字典的 key
  该方法将消耗 features 
  features:dict. k,v 为 str guid, feature
  encode_map: dict. k,v 为 str guid:int guid
  """
  encoded_features={}
  while features:
      guid, feature = features.popitem()
      guid = encode_map[guid]
      encoded_features[guid]=feature
  return encoded_features


# ========================= cowatch mining =========================
def get_one_list_of_cowatch(watched_guids):
  """ 解析一组 watched_guids 的 co-watch
  Args: 
    watched_guids: list of watched_guid in time order
    guids_set: set of guids
  """
  cowatch = []
  guids = []
  if len(watched_guids)>1:
    for i,guid in enumerate(watched_guids[:-1]):
      cowatch.append([watched_guids[i],watched_guids[i+1]])
      guids.append(guid)
  return cowatch, set(guids)


def get_all_cowatch(all_watched_guids):
  """ 解析所有 watched_guids 的 co-watch
  Args:
    all_watched_guids: list of watched_guids
  return：
    all_cowatch：list of cowatch guids
    all_guids: set of guid. all unique guids in all_cowatch
  """
  all_cowatch = []
  for watched_guids in tqdm(all_watched_guids):
    # TODO 这里可以用多线程
    cowatch,guids = get_one_list_of_cowatch(watched_guids)
    all_cowatch.extend(cowatch)
  return all_cowatch


def yield_all_cowatch(all_watched_guids):
  """ 解析所有 watched_guids 的 co-watch, 生成器版
  Args:
    all_watched_guids: list of watched_guids
  """
  for watched_guids in all_watched_guids:
    cowatch,guids = get_one_list_of_cowatch(watched_guids)
    yield cowatch


# ========================= aggregate cowatch =========================
def get_cowatch_graph(guids, all_cowatch):
  """统计所有观看历史，返回 co-watch 无向图，用于筛选 co-watch pair
  TODO: 过滤 co-watch 少于固定数量的 pair，过滤自己与自己组成的 pair
  NOTE: 运行在内存上，用矩阵存储和操作
  Args:
    guids: list of int.
    all_cowatch: list of cowatch(pair of guids)
  Retrun:
    cowatch_graph: dict
  """
  pass


def select_cowatch(cowatch_graph, threshold):
  """ 使用 cowatch_graph 丢弃低于 threshold 的 cowatched pair
  Args:
    cowatch_graph
    threshold
  Return:
    all_cowatch
  """
  pass


def arrays_to_dict(array_1d,array_2d):
  """ 将 1darray 和 array_2d 组合成一个字典
  要求两个array行数相同，array_1d.shape[0] == array_2d.shape[0]
  可用于生成 {guid:feature} 的 feature dict
  Args:
    array_1d: 1-D array-like 
    array_2d: 2-D ndarray
  Return:
    dict. key 为 array_1d 的元素，value 为 array_2d 的元素
  """
  return dict(zip(array_1d, array_2d))


# ========================= triplet mining =========================
def yield_negative_guid(guids):
  """循环输出随机采样样本，作为负样本"""
  neg_guids = copy.deepcopy(guids)
  np.random.shuffle(neg_guids)
  for neg_guid in cycle(neg_guids):
    yield neg_guid


def combine_cowatch_neg(cowatch, neg_iter):
  """为 cowatch 挑选 negative, 返回 triplet
  Args:
    cowatch: list of two guid
    neg_iter: iterator of random guids which is from the keys of features
  Retrun:
    triplet: the triplet is list of guids, the shape is (3, guid_size)
  """
  anchor_guid = cowatch[0]
  pos_guid = cowatch[1]
  neg_guid = neg_iter.__next__()
  while neg_guid in cowatch:
    neg_guid = neg_iter.__next__()
  triplet = [anchor_guid, pos_guid, neg_guid]
  return triplet


def mine_triplets(all_cowatch, features):
  """Get triplets for training model.
  A triplet contains an anchor, a positive, and a negative. Select 
  co-watch pair as anchor and positive, randomly sample a negative.
  NOTE: It is assumed that all guid can be retrieved in features.

  Args:
    all_cowatch: list of co-watch pair(list of guids)
    features: dict of features vector, the key is guid, 
              the value is feature string
    return_features: boolean. control the element in triplets
  Retrun:
    triplets: list of triplets.  triplet is list of 3 guid,
            [anchor_guid, pos_guid, neg_guid]
  """
  if not isinstance(all_cowatch,list) and not isinstance(features,dict):
    logging.error("Invalid arguments. Type should be list, dict instead of"+
      str(type(all_cowatch))+str(type(features)))
    return None
  guids=list(features.keys())
  neg_iter = yield_negative_guid(guids)
  # 初始化
  triplets = []
  # TODO 这里可以用多线程
  for cowatch in tqdm(all_cowatch):
    triplet = combine_cowatch_neg(cowatch, neg_iter)
    triplets.append(triplet)
  return triplets
  

# ========================= translate =========================
def lookup(batch_triplets, features):
  """Trans guids to features
  Arg:
    batch_triplets: 2-D ndarray of guid_triplets, guids are bytes after sess.run
      the shape is (batch_size, 3)
    features: dict of ndarray. keys are unicode
  Return:
    feature_triplets: 3-D ndarray of feature_triplets, the shape is (batch_size,
      3,1500)
  """
  triplets = []
  for _, guid_triplet in enumerate(batch_triplets):
    try:
      triplet = []
      for _, guid in enumerate(guid_triplet):
        guid=bytes.decode(guid) 
        triplet.append(features[guid])
      triplet = np.array(triplet)
      triplets.append(triplet)
    except Exception as e:
      logging.warning("lookup failed warning:"+str(e))
      continue
  feature_triplets = np.array(triplets)
  return feature_triplets

