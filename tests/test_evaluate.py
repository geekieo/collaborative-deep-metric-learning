# -*- coding: utf-8 -*-
''' 模型验证代码测试
@Description: 
@Date: 2019-07-10 17:31:26
@Author: Weng Jingyu
'''
# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from evaluate import l2_normalize
from evaluate import Evaluation
from online_data import load_cowatches

# eval_cowatches = load_cowatches('cdml_1/cowatches.eval')

embeddings = np.random.random((700000,256))
embeddings = l2_normalize(embeddings, axis=-1, order=2)
cowatches = np.random.randint(low=0,high=100000,size=(100000,2)) 

evaluater = Evaluation(embeddings, cowatches)
eval_dist = evaluater.mean_dist(evaluater.features, evaluater.cowatches)
print(eval_dist)