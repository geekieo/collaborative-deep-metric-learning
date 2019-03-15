# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
from tensorflow import logging
from online_data import read_features_txt
from online_data import trans_features_to_json
from online_data import read_features_json
from online_data import read_watched_guids
from online_data import get_unique_watched_guids
from online_data import get_watched_features

logging.set_verbosity(logging.DEBUG)

class Cache():
  def __init__(self):
    self.features={}
    self.all_watched_guids=[]
    self.unique_watched_guids=[]
    self.watched_feature={}

cache = Cache()

def test_read_features_txt():
  features = read_features_txt('features.txt')
  assert len(features)==254
  feature = list(features.values())[-1]
  elements = feature.split(',')
  assert len(elements)==1500
  cache.features=features


def test_trans_features_to_json():
  trans_features_to_json('features.txt', 'features.json')
  

def test_read_features_json():
  from_txt = read_features_txt('features.txt')
  from__json = read_features_json('features.json')
  assert from_txt.__eq__(from_txt)


def test_read_watched_guids():
  all_watched_guids = read_watched_guids('watched_guids.txt')
  assert all_watched_guids[0]== [
    '080be60a-7bac-43fb-af7a-9d55a3c72db3',
    '7f498113-5883-4e8e-b852-dd54fe9d283b',
    '3fc0540f-e79d-48d8-8902-8062bf498b35',
    '27f4acd4-e881-4cee-b4fa-4fce9a51a8af',
    'af801030-49ac-40fc-ba56-e8b5fc94a63c',
    '7e6840f6-569b-44ac-94bb-f685f7cde23f',
    'b8d18517-b61b-4b1a-b321-d999cc66c7af',
    'bcbc6cfd-4407-40be-bf04-d724d1da26b3']
  cache.all_watched_guids = all_watched_guids


def test_get_unique_watched_guids():
  unique_watched_guids = get_unique_watched_guids(cache.all_watched_guids)
  assert len(unique_watched_guids)==251
  cache.unique_watched_guids = unique_watched_guids


def test_get_watched_features():
  watched_feature = get_watched_features(cache.unique_watched_guids, cache.features)
  assert len(watched_feature)==251
  cache.watched_feature = watched_feature


if __name__ == "__main__":
  # test_read_features_txt()
  # test_trans_features_to_json()
  # test_read_features_json()
  # test_read_watched_guids()
  # test_get_unique_watched_guids()
  test_get_watched_features()
