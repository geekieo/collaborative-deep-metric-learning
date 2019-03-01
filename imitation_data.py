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


def gen_unique_id_array(low, high, size, dtype=None):
  """生成元素唯一的一维整数随机数组
  Args:
    low: int
    high: int
    size: int
    dtype:dtype, optional. The default value is ‘np.int’.
      option: np.bytes_
  Return:
    ids: ndarray of ints
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
  if dtype:
    ids = ids.astype(dtype)
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
  Args:
    ids: ndarray, [num_id]
    features: ndarray, [num_id, size_feature]
  Return:
    features: the first column is the id
  """
  ids = ids[:, np.newaxis] #1维转列向2维
  # ids = ids.astype(np.int)
  features = np.hstack((ids,features)) #NOTE:如果 ids.dtype 是 np.bytes_ 速度会很慢
  return features


def combine_feature_id(features, ids):
  """ 组合 id_array 和 feature_array
  ids 的长度需等于 feature 的行数
  Args:
    features: ndarray, [num_id, size_feature]
    ids: ndarray, [num_id]
  Return:
    features: the last column is the id
  """
  ids = ids[:, np.newaxis] #1维转列向2维
  # ids = ids.astype(np.int)
  features = np.hstack((features,ids)) #NOTE:如果 ids.dtype 是 np.bytes_ 速度会很慢
  return features


def gen_single_watched_guids(guids, low, high):
  """生成一条数量随机的 cowatch 数据
  Args:
    guids: 1-D array-like, list of bytes.
    low: int.
    high: int.
  Return:
    single_watched_guids: list of bytes. [size_single_watched]
  """
  size_single_watched = random.randint(low, high)
  single_watched_guids = np.random.choice(guids, size_single_watched)
  single_watched_guids = single_watched_guids.tolist()
  return single_watched_guids

def gen_watched_guids(guids, num_cowatch, low=2, high=30):
  """ 在 guids 中随机挑选随机数目的 guid
  Args:
    guids: ndarray of bytes
    num_cowatch: int. 
    low: int.Lowest (signed) integer to be drawn from the distribution 
    high: int, optional. If provided, one above the largest (signed) integer to be drawn from the distribution
  Return:
    watched_guids: list of single_watched_guids, [num_cowatch].
  """
  watched_guids = []
  for _ in range(num_cowatch):
    single_watched_guids = gen_single_watched_guids(guids, low, high)
    watched_guids.append(single_watched_guids)
  return watched_guids



def run():
  uids = gen_unique_id_array(low=1, high=num_uid*2, size=num_uid)
  guids = gen_unique_id_array(low=1, high=num_guid*2, size=num_guid)
  features = gen_features(num_feature=num_guid, feature_size=feature_size)

if __name__ == "__main__":
  run()
