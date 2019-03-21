# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np

from utils import exe_time
from imitation_data import num_uid, num_guid, feature_size
from imitation_data import gen_unique_id_array
from imitation_data import gen_watched_guids
from imitation_data import gen_all_watched_guids
from imitation_data import gen_features
from parse_data import get_guids_index
from parse_data import get_one_list_of_cowatch
from parse_data import get_all_cowatch
from parse_data import yield_all_cowatch
from parse_data import arrays_to_dict
from parse_data import yield_negative_guid
from parse_data import mine_triplets
from parse_data import lookup
from online_data import get_triplets

def test_get_guids_index():
  guids = gen_unique_id_array(low=1, high=3*2, size=3, dtype=np.bytes_)
  guids_index = get_guids_index(guids)
  print(guids_index)


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


def test_arrays_to_dict():
  guids = gen_unique_id_array(low=1, high=num_guid, size=num_guid, dtype=np.bytes_)
  features = gen_features(num_feature=num_guid, feature_size=feature_size)
  feature_dict = exe_time(arrays_to_dict)(guids, features)
  for k,v in feature_dict.items():
    print(k, ":\t", v)
    print(type(k), ":\t", type(v))
    break


def test_yield_negative_guid():
  guids = [1,2]
  negative_iter = yield_negative_guid(guids)
  neg1 = negative_iter.__next__()
  neg2 = negative_iter.__next__()
  neg3 = negative_iter.__next__()
  print(neg1, neg2, neg3)
  assert neg3 == 2 or neg3 == 1


def test_mine_triplets():
  guids = gen_unique_id_array(low=1, high=num_guid, size=num_guid, dtype=np.bytes_)
  watched_guids = gen_all_watched_guids(guids,num_cowatch=300,low=2,high=100)
  all_cowatch = get_all_cowatch(watched_guids)
  features_vectors = gen_features(num_feature=num_guid, feature_size=feature_size)
  features=arrays_to_dict(guids,features_vectors)

  triplet = mine_triplets(all_cowatch, features)
  print(type(triplet), len(triplet), triplet[0])


def test_lookup():
  triplets, features = get_triplets(watch_file="watched_guids.txt",
                                    feature_file="features.txt")
  triplets = lookup(triplets,features)
  print(triplets)
  print(triplets.shape)


if __name__ == "__main__":
  # test_get_one_list_of_cowatch()
  # test_yield_all_cowatch()
  # test_arrays_to_dict()
  # test_yield_negative_guid()
  # test_mine_triplets()
  test_lookup()