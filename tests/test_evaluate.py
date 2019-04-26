# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from evaluate import l2_normalize
from evaluate import mean_dist

embeddings = np.random.random((10000,256))
embeddings = l2_normalize(embeddings, axis=-1, order=2)
cowatches = np.random.randint(low=0,high=10000,size=(1000,2)) 
dist = mean_dist(embeddings, cowatches)
print(dist)