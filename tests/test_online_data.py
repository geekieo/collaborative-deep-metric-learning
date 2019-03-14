# -*- coding:utf-8 -*-
import sys
sys.path.append("..")

from online_data import read_features_txt
from online_data import trans_features_to_json
from online_data import read_features_json



def test_read_features_txt():
  features = read_features_txt('features.txt')
  assert list(features.keys())==[
    'c0115fd1-1f1b-4ee4-a596-70ae95f0e219',
    '4f2a3543-8c42-4314-bfe0-de59fb0ee694',
    '8b226bb2-c097-480f-8f22-2208c1a6fcbe']
  feature = list(features.values())[0]
  elements = feature.split(',')
  assert len(elements)==1500


def test_trans_features_to_json():
  trans_features_to_json('features.txt', 'features.json')
  
def test_read_features_json():
  from_txt = read_features_txt('features.txt')
  from__json = read_features_json('features.json')
  assert from_txt.__eq__(from_txt)



if __name__ == "__main__":
  test_read_features_json()