# -*- coding:utf-8 -*-
""" Parse watched guids to triplets """


def get_one_list_of_cowatch(watched_guids):
  """ 解析一组 watched_guids 的 co-watch
  Args: 
    watched_guids: list of watched_guid in time order
  """
  cowatch = []
  if len(watched_guids)>1:
    for i,guid in enumerate(watched_guids[:-1]):
      cowatch.append([watched_guids[i],watched_guids[i+1]])
  return cowatch

def get_all_cowatch(all_watched_guids):
  """ 解析所有 watched_guids 的 co-watch
  Args:
    all_watched_guids: list of watched_guids
  """
  all_cowatch = []
  for watched_guids in all_watched_guids:
    cowatch = get_one_list_of_cowatch(watched_guids)
    all_cowatch.extend(cowatch)
  return all_cowatch


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


def threshold_filter(cowatch_graph, threshold):
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


def get_triplet(guids, features, all_cowatch):
  """Get triplets for training model.
  A triplet contains an anchor, a positive, and a negative. Select 
  co-watch pair as anchor and positive, randomly sample a negative.
  Args:
    guids: ndarray of int. the ids of features
    features: ndarray of features vector, which is a ndarray of 1500 float)
    all_cowatch: list of co-watch pair(pair of guids)
  Retrun:
    triplets: ndarray of [anchor feature, positive feature, negative feature]
  """
  feature_dict = arrays_to_dict(guids, features)
  for cowatch in all_cowatch:
    


  # return triplets

  

