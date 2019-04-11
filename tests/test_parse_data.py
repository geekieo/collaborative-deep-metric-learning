# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
import copy

from utils import exe_time
from online_data import read_features_txt
from online_data import read_watched_guids
from online_data import get_triplets
from imitation_data import num_uid, num_guid, feature_size
from imitation_data import gen_unique_id_array
from imitation_data import gen_watched_guids
from imitation_data import gen_all_watched_guids
from imitation_data import gen_features
from imitation_data import arrays_to_dict
from parse_data import get_unique_watched_guids
from parse_data import filter_features
from parse_data import filter_watched_guids
from parse_data import encode_guids
from parse_data import encode_base_features
from parse_data import encode_features
from parse_data import encode_all_watched_guids
from parse_data import get_one_list_of_cowatch
from parse_data import get_all_cowatch
from parse_data import yield_all_cowatch
from parse_data import yield_negative_guid
from parse_data import mine_triplets
from parse_data import get_cowatch_graph
from parse_data import select_cowatch
from parse_data import lookup


def test_get_unique_watched_guids():
  all_watched_guids = read_watched_guids('watched_guids.txt')
  unique_watched_guids = get_unique_watched_guids(all_watched_guids)
  assert len(unique_watched_guids)==251


def test_filter_features():
  features = read_features_txt('visual_features.txt')
  assert len(features)==254
  all_watched_guids = read_watched_guids('watched_guids.txt')
  # 制造不和 all_watched_guids 对应的 features
  print(type(all_watched_guids[0][1]))
  features.pop(all_watched_guids[0][1])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  # print(len(watched_feature))
  assert len(features)==250
  assert len(no_feature_guids)==1


def test_filter_watched_guids():
  features_ori = read_features_txt("visual_features.txt")
  assert len(features_ori)==254
  all_watched_guids = read_watched_guids('watched_guids.txt')

  features = copy.deepcopy(features_ori)
  # print(all_watched_guids[0][0])
  features.pop(all_watched_guids[0][0])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  filtered_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)
  assert len(all_watched_guids)==13
  assert len(filtered_watched_guids)==13
  # print(filtered_watched_guids[0])
  assert len(filtered_watched_guids[0])==7

  features = copy.deepcopy(features_ori)
  features.pop(all_watched_guids[0][1])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  filtered_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)
  assert len(all_watched_guids)==13
  assert len(filtered_watched_guids)==13
  assert len(filtered_watched_guids[0])==6
  
  features = copy.deepcopy(features_ori)
  features.pop(all_watched_guids[0][2])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  filtered_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)
  assert len(all_watched_guids)==13
  assert len(filtered_watched_guids)==14
  assert len(filtered_watched_guids[0])==2
  
  # 测试后半部分
  features = copy.deepcopy(features_ori)
  features.pop(all_watched_guids[0][5])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  filtered_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)
  assert len(all_watched_guids)==13
  assert len(filtered_watched_guids)==14
  assert len(filtered_watched_guids[0])==5
  assert len(filtered_watched_guids[1])==2

  features = copy.deepcopy(features_ori)
  features.pop(all_watched_guids[0][6])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  filtered_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)
  assert len(all_watched_guids)==13
  assert len(filtered_watched_guids)==13
  assert len(filtered_watched_guids[0])==6

  features = copy.deepcopy(features_ori)
  features.pop(all_watched_guids[0][7])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  filtered_watched_guids = filter_watched_guids(all_watched_guids, no_feature_guids)
  assert len(all_watched_guids)==13
  assert len(filtered_watched_guids)==13
  assert len(filtered_watched_guids[0])==7


def test_encode_guids():
  guids = gen_unique_id_array(low=8, high=8*2, size=5, dtype=np.bytes_)
  guids = guids.tolist()
  guids.append(guids[0])
  # print(guids)
  guids = iter(guids)
  encode_map, decode_map = encode_guids(guids)
  for guid in guids:
    assert decode_map[encode_map[guid]]==guid


def test_encode_features():
  features = read_features_txt('visual_features.txt')
  features_ori = copy.deepcopy(features)
  encode_map, decode_map = encode_base_features(features)
  encoded_features = encode_features(features, decode_map)
  assert encoded_features[0] == features_ori[decode_map[0]]


def test_encode_all_watched_guids():
  features = read_features_txt('visual_features.txt')
  encode_map, decode_map = encode_base_features(features)
  all_watched_guids = read_watched_guids('watched_guids.txt')
  encoded_all_watch_guids = encode_all_watched_guids(all_watched_guids, encode_map)
  print(encoded_all_watch_guids)
  assert len(encoded_all_watch_guids)==13
  assert encoded_all_watch_guids[0] is not None


def test_get_one_list_of_cowatch():
  watch_guids = gen_unique_id_array(low=1, high=1000*2, size=1000, dtype=np.bytes_)
  cowatch,guids = get_one_list_of_cowatch(watch_guids)
  assert len(cowatch)==len(watch_guids)-1
  print(cowatch[:5])
  assert len(guids)==999

  cowatch,guids = get_one_list_of_cowatch([1,2])
  assert len(cowatch)==1
  print(cowatch)

  cowatch,guids = get_one_list_of_cowatch([1])
  assert len(cowatch)==0
  print(cowatch)


def test_get_all_cowatch():
  guids = gen_unique_id_array(low=1, high=10*2, size=10)
  watched_guids = gen_all_watched_guids(guids,num_cowatch=300,low=2,high=100)
  all_cowatch = exe_time(get_all_cowatch)(watched_guids)
  print(len(all_cowatch), all_cowatch[:5])

  watched_guids = [[0], [1,2], [3,4,5,6], [], [7,8,9]]
  all_cowatch = get_all_cowatch(watched_guids)
  assert len(all_cowatch)==6
  print(all_cowatch)


def test_yield_all_cowatch():
  guids = gen_unique_id_array(low=1, high=10*2, size=10)
  watched_guids = gen_all_watched_guids(guids,num_cowatch=300,low=2,high=10)
  cowatch = yield_all_cowatch(watched_guids)
  print(cowatch.__next__())
  print(cowatch.__next__())


def test_yield_negative_guid():
  guids = [1,2]
  negative_iter = yield_negative_guid(guids)
  neg1 = negative_iter.__next__()
  neg2 = negative_iter.__next__()
  neg3 = negative_iter.__next__()
  print(neg1, neg2, neg3)
  assert neg3 == 2 or neg3 == 1


def test_mine_triplets():
  guids = gen_unique_id_array(low=0, high=num_guid, size=num_guid, dtype=np.int)
  watched_guids = gen_all_watched_guids(guids,num_cowatch=300,low=2,high=100)
  all_cowatch = get_all_cowatch(watched_guids)
  features_vectors = gen_features(num_feature=num_guid, feature_size=feature_size)
  # features=arrays_to_dict(guids,features_vectors)

  triplet = mine_triplets(all_cowatch, features_vectors)
  print(type(triplet), len(triplet), triplet[0])


def test_get_cowatch_graph():
  watched_guids = [[0],[], [3, 8, 2, 2, 8], [0, 4, 5, 1, 7, 1], [1, 0, 7, 3, 8, 5], [5, 5, 8, 6],
                   [2, 1, 7, 3], [3, 2, 5, 1, 4, 0, 1, 7, 7, 0, 2, 8, 1]]
  all_cowatch = get_all_cowatch(watched_guids)
  print(all_cowatch)
  graph = get_cowatch_graph(all_cowatch)
  print(graph)


def test_selest_cowatch():
  watched_guids = [[0],[], [3, 8, 2, 2, 8], [0, 4, 5, 1, 7, 1], [1, 0, 7, 3, 8, 5], [5, 5, 8, 6],
                   [2, 1, 7, 3], [3, 2, 5, 1, 4, 0, 1, 7, 7, 0, 2, 8, 1]]
  all_cowatch = get_all_cowatch(watched_guids)
  graph = get_cowatch_graph(all_cowatch)
  new_all_cowatch = select_cowatch(graph, threshold=3)
  print('graph', graph)
  print('all_cowatch',all_cowatch)
  print('new_all_cowatch',new_all_cowatch)

@DeprecationWarning
def test_lookup():
  triplets, features = get_triplets(watch_file="watched_guids.txt",
                                    feature_file="visual_features.txt")
  guid_triplets = tf.constant(triplets)
  with tf.Session() as sess:
    guid_triplets_np = sess.run(guid_triplets)
  triplets = lookup(guid_triplets_np, features)
  print(triplets)
  print(triplets.shape)


if __name__ == "__main__":
  # test_get_unique_watched_guids()
  # test_filter_features()
  # test_filter_watched_guids()
  # test_encode_guids()
  # test_encode_features()
  # test_encode_all_watched_guids()
  # test_get_one_list_of_cowatch()
  # test_get_cowatch_graph()
  # test_selest_cowatch()
  # test_yield_all_cowatch()
  # test_yield_negative_guid()
  test_mine_triplets()
  # test_lookup()