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
  """生成元素唯一的一维整数随机数组,用于生成 uids 和 guids
  Args:
    low: int
    high: int
    size: int
    dtype: dtype, optional. The default value is ‘np.int’.
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
  ids = random.sample(random_list, size)  #唯一的二维整数随机列表
  ids = np.array(ids)                     #唯一的二维整数随机数组
  ids = ids.flatten()                     #唯一的一维整数随机数组
  if dtype:
    ids = ids.astype(dtype)
  return ids


def gen_features(num_feature, feature_size, decimals=8):
  """ 生成2D数组，元素范围在[-1,1]之间
  Args:
    num_feature: int
    feature_size: int
    decimals: int. 小数；精度
  Return:
    features: 2d array. The shape is [num_feature, feature_size]
  """
  shape = (num_feature, feature_size)
  features = np.random.random(shape)
  features = np.around(features, decimals)
  return features


def gen_watched_guids(guids, low, high):
  """生成一条数量随机的 watched guids
  Args:
    guids: 1-D array-like, list of bytes.
    low: int.
    high: int.
  Return:
    watched_guids: list of bytes. [size_single_watched]
  """
  size_watched = random.randint(low, high)
  watched_guids = np.random.choice(guids, size_watched)
  watched_guids = watched_guids.tolist()
  return watched_guids


def gen_all_watched_guids(guids, num_cowatch, low=2, high=30):
  """ 在 guids 中随机挑选随机数目的 guid
  Args:
    guids: ndarray of bytes
    num_cowatch: int. 
    low: int.Lowest (signed) integer to be drawn from the distribution 
    high: int, optional. If provided, one above the largest (signed) integer to be drawn from the distribution
  Return:
    all_watched_guids: list of watched_guids, [num_cowatch].
  """
  all_watched_guids = []
  for _ in range(num_cowatch):
    watched_guids = gen_watched_guids(guids, low, high)
    all_watched_guids.append(watched_guids)
  return all_watched_guids


def gen_triplets(batch_size, feature_size):
  shape = [batch_size, 3, feature_size]
  triplets = gen_features(num_feature=shape[0]*shape[1],
                          feature_size=shape[2])
  triplets = np.reshape(triplets, shape)
  return triplets


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
