# -*- coding:utf-8 -*-
""" Parse watched guids to triplets """
import numpy as np
from itertools import cycle
import copy

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
  """
  all_cowatch = []
  all_guids = {}
  for watched_guids in all_watched_guids:
    cowatch,guids = get_one_list_of_cowatch(watched_guids)
    all_cowatch.extend(cowatch)
    all_guids = all_guids|guids
  return all_cowatch, list(all_guids)


def yield_all_cowatch(all_watched_guids):
  """ 解析所有 watched_guids 的 co-watch, 生成器版
  Args:
    all_watched_guids: list of watched_guids
  """
  for watched_guids in all_watched_guids:
    cowatch = get_one_list_of_cowatch(watched_guids)
    yield cowatch


def get_guids_index(guids):
  """ guid 索引字典 
  用于查找 guid 在 guids 中的索引
  """
  return {k: v for v, k in enumerate(guids)}
  

def get_cowatch_graph(guids, all_cowatch):
  """统计所有观看历史，返回 co-watch 无向图，用于筛选 co-watch pair
  TODO: 过滤 co-watch 少于固定数量的 pair，过滤自己与自己组成的 pair
  NOTE: 运行在内存上，用矩阵存储和操作
  Args:
    guids: list of int.
    all_cowatch: list of cowatch(pair of guids)
  Retrun:
    cowatch_graph
  """
  pass


def cowatch_graph_filter(cowatch_graph, threshold):
  """ 使用 cowatch_graph 过滤低于 threshold 的 cowatch 
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


def yield_negative_guid(guids):
  """循环输出随机采样样本，作为负样本"""
  neg_guids = copy.deepcopy(guids)
  np.random.shuffle(neg_guids)
  for neg_guid in cycle(neg_guids):
    yield neg_guid


def get_triplet(guids, all_cowatch, features):
  """Get triplets for training model.
  A triplet contains an anchor, a positive, and a negative. Select 
  co-watch pair as anchor and positive, randomly sample a negative.

  Args:
    guids: ndarray of int. the ids of features
    all_cowatch: list of co-watch pair(list of guids)
    features: ndarray of features vector(ndarray of 1500 float)
  Retrun:
    triplets: ndarray of [anchor feature, positive feature, negative feature]
  """
  feature_dict = arrays_to_dict(guids, features)
  neg_iter = yield_negative_guid(guids)
  triplets = []
  for cowatch in all_cowatch:
    anchor_guid = cowatch[0]
    anchor = feature_dict[anchor_guid]
    pos_guid = cowatch[1]
    pos = feature_dict[pos_guid]
    neg_guid = neg_iter.__next__()
    while neg_guid in cowatch:
      neg_guid = neg_iter.__next__()
    neg = feature_dict[neg_guid]
    triplet = [anchor, pos, neg]
    triplets.append(triplet)
  return np.array(triplets)
  

