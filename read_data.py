# -*- coding:utf-8 -*-
import json
import tensorflow as tf

data_file = "sample.json"
data = []
with open(data_file, 'r',encoding='utf-8') as f:
  try:
    while True:
        line = f.readline()
        if line:
            sample = json.loads(line)
            data.append()
            print(sample)
        else:
            break
  except:
      f.close()
