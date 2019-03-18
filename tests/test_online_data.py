# -*- coding:utf-8 -*-
""" watch_guids.txt 中的 guid 在 features.txt 中均有对应 feature"""
import sys
sys.path.append("..")
import copy
from online_data import read_features_txt
from online_data import trans_features_to_json
from online_data import read_features_json
from online_data import read_watched_guids
from online_data import get_unique_watched_guids
from online_data import filter_features
from online_data import filter_watched_guids
from online_data import get_triplets



def test_read_features_txt():
  features = read_features_txt('features.txt')
  assert len(features)==254
  feature = list(features.values())[-1]
  elements = feature.split(',')
  assert len(elements)==1500


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


def test_get_unique_watched_guids():
  all_watched_guids = read_watched_guids('watched_guids.txt')
  unique_watched_guids = get_unique_watched_guids(all_watched_guids)
  assert len(unique_watched_guids)==251

def test_filter_features():
  features = read_features_txt('features.txt')
  assert len(features)==254
  all_watched_guids = read_watched_guids('watched_guids.txt')
  # 制造不和 all_watched_guids 对应的 features
  features.pop(all_watched_guids[0][1])
  features, no_feature_guids = filter_features(features, all_watched_guids)
  # print(len(watched_feature))
  assert len(features)==250
  assert len(no_feature_guids)==1


def test_filter_watched_guids():

  features_ori = read_features_txt("features.txt")
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

def test_get_triplets():
  triplets = get_triplets(watch_file="watched_guids.txt", feature_file="features.txt")
  assert len(triplets)==240

def test_get_triplets_real():
  try:
    triplets = get_triplets(watch_file="watched_video_ids", feature_file="video_guid_inception_feature.txt")
  except Exception as e:
    print(e)

if __name__ == "__main__":
  # test_read_features_txt()
  # test_trans_features_to_json()
  # test_read_features_json()
  # test_read_watched_guids()
  # test_get_unique_watched_guids()
  # test_filter_features()
  # test_filter_watched_guids()
  # test_get_triplets()
  test_get_triplets_real()
