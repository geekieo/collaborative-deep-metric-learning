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
from parse_data import get_guids_index
from parse_data import get_one_list_of_cowatch
from parse_data import get_all_cowatch
from parse_data import yield_all_cowatch

def test_get_guids_index():
  guids = gen_unique_id_array(low=1, high=3*2, size=3, dtype=np.bytes_)
  guids_index = get_guids_index(guids)
  print(guids_index)


def test_get_one_list_of_cowatch():
  watch_guids = gen_unique_id_array(low=1, high=1000*2, size=1000, dtype=np.bytes_)
  cowatch = get_one_list_of_cowatch(watch_guids)
  assert len(cowatch)==len(watch_guids)-1
  print(cowatch[:5])

  cowatch = get_one_list_of_cowatch([1,2])
  assert len(cowatch)==1
  print(cowatch)

  cowatch = get_one_list_of_cowatch([1])
  assert len(cowatch)==0
  print(cowatch)


def test_get_all_cowatch():
  guids = gen_unique_id_array(low=1, high=10*2, size=10)
  watched_guids = gen_all_watched_guids(guids,num_cowatch=300,low=2,high=100)
  all_cowatch = exe_time(get_all_cowatch)(watched_guids)
  print(all_cowatch[:5])
  print(len(all_cowatch))

  watched_guids = [[0], [1,2], [3,4,5,6], [], [7,8,9]]
  cowatch = get_all_cowatch(watched_guids)
  assert len(cowatch)==6
  print(cowatch)

def test_yield_all_cowatch():
  guids = gen_unique_id_array(low=1, high=10*2, size=10)
  watched_guids = gen_all_watched_guids(guids,num_cowatch=300,low=2,high=10)
  cowatch = yield_all_cowatch(watched_guids)
  print(cowatch.__next__())
  print(cowatch.__next__())

if __name__ == "__main__":
  test_yield_all_cowatch()