# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from utils import exe_time
from imitation_data import num_uid, num_guid, feature_size
from imitation_data import gen_unique_id_array
from imitation_data import gen_features
from imitation_data import gen_watched_guids
from imitation_data import gen_all_watched_guids


def test_gen_unique_id_array():
  uids = gen_unique_id_array(low=1, high=num_uid*2, size=num_uid)
  guids = gen_unique_id_array(low=1, high=num_guid*2, size=num_guid)
  print(uids.shape, sys.getsizeof(uids), type(uids[0]),uids)
  print(guids.shape, sys.getsizeof(guids), type(guids[0]),guids)


def test_gen_features(num_feature=num_guid, feature_size=feature_size):
  features = gen_features(num_feature=num_guid, feature_size=feature_size)
  print(features.shape, sys.getsizeof(features), type(features[0]),features)


def test_gen_watched_guids():
  guids = gen_unique_id_array(low=1, high=num_guid*2, size=num_guid, dtype=np.bytes_)
  watched_guids = gen_watched_guids(guids=guids, low=2, high=30)
  print(watched_guids)


def test_gen_all_watched_guids():
  guids = gen_unique_id_array(low=1, high=num_guid*2, size=num_guid, dtype=np.bytes_)
  watched_guids = gen_all_watched_guids(guids, num_cowatch=1000, low=2, high=30)
  print(type(watched_guids),type(watched_guids[0]),watched_guids[:3])


if __name__ == "__main__":
  # test_gen_unique_id_array()
  test_gen_features()
  # test_gen_watched_guids()
  # test_gen_all_watched_guids()
