# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import 

import models
import 
from utils import find_class_by_name
from train import Trainer


train_dir="."
model = find_class_by_name(FLAGS.model, [models])()

class test_Trainer():
  trainer = Trainer()
