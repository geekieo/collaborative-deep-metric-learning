# -*- coding:utf-8 -*-
""" watch_guids.txt 中的 guid 在 visual_features.txt 中均有对应 feature"""
import sys
sys.path.append("..")
import copy
import sys

from online_data import read_features_txt
# from online_data import read_features_json
from online_data import read_watched_guids
from online_data import get_triplets
from online_data import gen_trining_data



def test_read_features_txt():
  features = read_features_txt('visual_features.txt')
  assert len(features)==254
  feature = list(features.values())[-1]
  assert len(feature)==1500
  print(feature)
  print(type(feature))


# def test_read_features_json():
#   features = read_features_json('features.json')
#   print(features['0'])
#   print(len(features))


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


def test_get_triplets():
  triplets,features,encode_map,decode_map = get_triplets(
                            watch_file="watched_guids.txt",
                            feature_file="visual_features.txt")
  shape = (len(triplets),len(triplets[0]))
  print(shape)
  print(triplets)
  # print(features)
  # print(encode_map)
  # print(decode_map)


def test_read_features_txt_real():
  try:
    features = read_features_txt('/data/wengjy1/video_guid_inception_feature.txt')
    print(sys.getsizeof(features))
    feature = list(features.values())[-1]
    assert len(elements)==1500
  except Exception as e:
    print(e)

def test_get_triplets_real():
  try:
    triplets, features = get_triplets(
      watch_file="/data/wengjy1/watched_video_ids",
      feature_file="/data/wengjy1/video_guid_inception_feature.txt")
  except Exception as e:
    print(e)

def test_gen_trining_data():
  gen_trining_data(watch_file="watched_guids.txt",
                   feature_file="visual_features.txt",
                   save_dir = '')


if __name__ == "__main__":
  # test_read_features_txt()
  # test_read_watched_guids()
  # test_get_triplets()
  # test_read_features_txt_real()
  # test_get_triplets_real()
  test_gen_trining_data()
