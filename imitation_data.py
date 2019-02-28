import numpy as np 
import itertools
import random 
import sys
from utils import exe_time


# user
num_uid = 30000
# video
num_guid = 10000
feature_size = 1500 


def gen_unique_id_array(low, high, size,to_bytes=False):
  """生成元素唯一的一维整数随机数组
  Args:
    low: int
    high: int
    size: int
  Return:
    ids: 1d array
  """
  if low > high:
    raise("low > high")
  elif high-low+1 < size:
    raise("size is larger than high-low+1")
  elif size < 0:
    raise("size is negative")

  random_list = list(itertools.product(range(low, high+1))) #唯一的二维整数顺序列表
  ids = random.sample(random_list, size) #唯一的二维整数随机列表
  ids = np.array(ids)                   #唯一的二维整数随机数组
  ids = ids.flatten()                   #唯一的一维整数随机数组
  if to_bytes:
    ids = ids.astype(np.bytes_)
  return ids


def gen_features(num_feature, feature_size, decimals=8):
  """ 生成2D数组，元素范围在[-1,1]之间
  Args:
    num_feature: int
    feature_size: int
  Return:
    features: 2d array
  """
  shape = (num_feature, feature_size)
  features = np.random.random(shape)
  features = np.around(features, decimals)
  return features


def combine_id_feature(ids, features):
  """ 组合 id_array 和 feature_array
  ids 的长度需等于 feature 的行数
  Arg:
    ids: 1d array, [num_id]
    features: 2d array, [num_id, size_feature]
  Return:
    features: the first column is the id
  """
  ids = ids[:, np.newaxis] #1维转列向2维
  ids = ids.astype(np.int)
  features = np.hstack((ids,features)) #NOTE:如果 ids.dtype 是 np.bytes_ 速度会很慢
  return features

def run():
  uids = gen_unique_id_array(low=1, high=num_uid*2, size=num_uid)
  guids = gen_unique_id_array(low=1, high=num_guid*2, size=num_guid)
  features = gen_features(num_feature=num_guid, feature_size=feature_size)
  features = combine_id_feature(guids, features)

if __name__ == "__main__":
    run()


