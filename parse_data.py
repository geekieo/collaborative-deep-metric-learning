# -*- coding:utf-8 -*-
from functools import reduce

def get_guids_index(guids):
  """ guid 索引字典 
  用于查找 guid 在 guids 中的索引
  """
  return {k: v for v, k in enumerate(guids)}

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


def get_cowatch_graph(guids, all_watched_guids):
  """统计所有观看历史，返回 cowatch 无向图，用稀疏矩阵表示，用于筛选 cowatch
  NOTE: 如果显存不够，用 CPU 运行在内存上
  Args:
    guids: 
    all_watched_guids: 2-D ndarray 
  Retrun:
    cowatch_graph: SparseTensor
  """
  guids_size = len(guids)
  dense_shape = [guids_size, guids_size]

  guids_index = get_guids_index(guids)
  # for watched_guids in all_watched_guids:
    
  # indices = 
  # values = 
  



  


  
