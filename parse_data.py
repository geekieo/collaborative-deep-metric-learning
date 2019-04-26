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
import collections

logging.set_verbosity(logging.DEBUG)


# ========================= filter guids =========================
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
  删除 features 中用 all_watched_guids 没访问的 feature
  并找出无法获取到 feature 的 guid 
  Args:
    features: dict
    all_watched_guids: 2-D list. list of watched guids.
  Return:
    watched_features: dict
    no_feature_guids: list
  """
  unique_watched_guids = get_unique_watched_guids(all_watched_guids)
  no_feature_guids=[]
  watched_features=collections.OrderedDict()
  for guid in unique_watched_guids:
    try:
      watched_features[guid]=features.pop(guid)
    except KeyError as e:
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


# ========================= encoding guid =========================
def encode_base_features(features):
  """ 以 features 的 keys 为基准，把 guids 编码成有序的 index
  features:dict. k,v 为 str guid, feature
  """
  encode_map=collections.OrderedDict()
  decode_map=collections.OrderedDict()
  for index, guid in enumerate(features.keys()):
    if not guid in encode_map:
      encode_map[guid]=index
      decode_map[index]=guid
  return encode_map, decode_map


def encode_features(features, decode_map):
  """ 编码 features
  NOTE: 用 decode_map 编码 features
  将 features 按照 decode_map 的编码从0开始有序放进列表。
  decode_map 编码自 features，key 为从0开始有序的整数。
  该方法会消费原始 features 字典。
  args:
    features:dict. k,v 为 str guid, feature vector
    decode_map: OrderedDict. k,v 为 int guid: str guid
  return:
    array_feature: 
  """
  encoded_features = []
  for index, int_guid in enumerate(decode_map):
    str_guid = decode_map[index]
    encoded_features.append(features[str_guid])# list of 1500 float
  return np.asarray(encoded_features, dtype=np.float32)


def encode_watched_guids(watched_guids, encode_map):
  """ 编码一组 watched_guids 
  该方法会消费 watched_guids
  """
  encoded_watched_guids=[]
  while watched_guids:
    guid = watched_guids.pop()
    encoded_watched_guids.append(encode_map[guid])
  return encoded_watched_guids


def encode_all_watched_guids(all_watched_guids, encode_map):
  """ 编码全部 watched_guids
  该方法会消费 all_watched_guids
  """
  encoded_all_watch_guids=[]
  while all_watched_guids:
    watched_guids = all_watched_guids.pop()
    watched_guids = encode_watched_guids(watched_guids, encode_map)
    encoded_all_watch_guids.append(watched_guids)
  return encoded_all_watch_guids


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
  return:
    all_cowatch: list of cowatch guids
    all_guids: set of guid. all unique guids in all_cowatch
  """
  all_cowatch = []
  for watched_guids in all_watched_guids:
    cowatch,guids = get_one_list_of_cowatch(watched_guids)
    all_cowatch.extend(cowatch)
  np.random.shuffle(all_cowatch)
  return all_cowatch


def yield_all_cowatch(all_watched_guids):
  """ 解析所有 watched_guids 的 co-watch, 生成器版
  Args:
    all_watched_guids: list of watched_guids
  """
  for watched_guids in all_watched_guids:
    cowatch,guids = get_one_list_of_cowatch(watched_guids)
    yield cowatch


# ========================= select cowatch =========================
def get_cowatch_graph(cowatches):
  """统计所有观看历史，返回 co-watch 无向图，丢弃自己与自己组成的 pair
  NOTE 所有 co-watch 中的 guid 必须是 index.
  Args:
    guids: list of co-watch pair. the guid should be index.
  Retrun:
    graph: dict
  """
  graph = {}
  drop_count=0
  for i in range(len(cowatches) - 1, -1, -1): #倒序
    cowatch = cowatches[i]
    if not(isinstance(cowatch[0],int) and isinstance(cowatch[0],int)):
      rint('get_cowatch_graph: cowatch is not int',str(cowatch))
      cowatches.pop(i)
      continue
    # 有向转无向，并丢弃自己与自己组成的 co-watch pair
    if cowatch[0] < cowatch[1]:
      edge = str(cowatch[0])+','+str(cowatch[1])
    elif cowatch[0] > cowatch[1]:
      edge = str(cowatch[1])+','+str(cowatch[0])
    else:
      drop_count += 1
      cowatches.pop(i)
      continue
    # 统计边长
    if edge in graph:
      graph[edge] += 1
    else:
      graph[edge] = 1
  print('get_cowatch_graph: drop self pair: ', str(drop_count))
  return graph, cowatches

def select_cowatch(cowatch_graph, threshold, cowatches=None, unique=False):
  """ 使用 cowatch_graph 筛选高于 threshold 的 cowatched pair
  Args:
    cowatch_graph
    threshold
    cowatches: shuffled
  Return:
    selected_cowatches
  """
  threshold = int(threshold)
  selected_cowatches = []
  if unique:
    # 返回 cowatch 唯一的 cowatches. 从 graph 的 key 中提取 cowatch, 打乱后返回.
    for edge in cowatch_graph:
      if cowatch_graph[edge]>=threshold:
        cowatch = list(map(int, edge.split(',')))
        np.random.shuffle(cowatch)
        selected_cowatches.append(cowatch)
        np.random.shuffle(selected_cowatches)
  else:
    if cowatches is None:
      logging.error('cowatches is None')
      return cowatches
    if threshold <= 1:
      return cowatches
    # 返回存在重复 cowatch 的 cowatches
    for cowatch in cowatches:
      edge = str(cowatch[0])+','+str(cowatch[1]) if cowatch[0]<cowatch[1] else str(cowatch[1])+','+str(cowatch[0])
      try:
        if cowatch_graph[edge] < threshold:
          selected_cowatches.append(cowatch)
      except Exception as e:
        logging.warning('parse_data select_cowatch '+str(e))
    
  return selected_cowatches
  
# ========================= triplet mining =========================
def yield_negative_index(size, putback=False):
  """生成 [0,size-1] 的随机整数
  putback:有放回的抽取
  """
  if putback:
    while True:
      yield np.random.randint(0,size)
  else:
    indexes = list(range(size))
    np.random.shuffle(indexes)
    for neg_index in cycle(indexes):
      yield neg_index


def combine_cowatch_neg(cowatch, neg_iter):
  """为 cowatch 挑选 negative, 返回 triplet
  Args:
    cowatch: list of two guid index
    neg_iter: iterator of random guid index
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

@DeprecationWarning
def mine_triplets(all_cowatch, features):
  """Get triplets for training model.
  A triplet contains an anchor, a positive, and a negative. Select 
  co-watch pair as anchor and positive, randomly sample a negative.
  NOTE: It is assumed that all guid can be retrieved in features.

  Args:
    all_cowatch: list of co-watch pair(list of guids)
    features: 2-D ndarray of featuress, the index is guid, 
              the value is ndarray
  Retrun:
    triplets: list of triplets.  triplet is list of 3 guid,
            [anchor_guid, pos_guid, neg_guid]
  """
  if not isinstance(all_cowatch,list) and not isinstance(features,np.ndarray):
    logging.error("Invalid arguments. Type should be list, dict instead of"+
      str(type(all_cowatch))+str(type(features)))
    return None
  neg_iter = yield_negative_index(len(features), putback=True)
  # 初始化
  triplets = []
  # TODO 这里可以用多线程
  for cowatch in all_cowatch:
    triplet = combine_cowatch_neg(cowatch, neg_iter)
    triplets.append(triplet)
  return triplets
  

# ========================= translate =========================
@DeprecationWarning
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
      triplet = np.asarray(triplet)
      triplets.append(triplet)
    except Exception as e:
      logging.warning("lookup failed warning:"+str(e))
      continue
  feature_triplets = np.asarray(triplets)
  return feature_triplets

