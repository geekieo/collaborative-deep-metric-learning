# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from evaluate import l2_normalize
from evaluate import Evaluater

embeddings = np.random.random((700000,256))
embeddings = l2_normalize(embeddings, axis=-1, order=2)
cowatches = np.random.randint(low=0,high=100000,size=(100000,2)) 

evaluater = Evaluater(embeddings, cowatches)
eval_dist = evaluater.mean_dist(evaluater.eval_features, evaluater.eval_cowatches)
print(eval_dist)