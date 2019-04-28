# -*- encode:utf-8 -*-
import numpy as np
from parse_data import get_unique_watched_guids

def l2_normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class Evaluater():
  def __init__(self, features, cowatches):
    self.eval_features, self.eval_cowatches = self._rencode(features, cowatches)

  def _rencode(self, features, cowatches):
    unique_indexes = get_unique_watched_guids(cowatches)
    print("Evaluater._rencode unique_indexes num: ",len(unique_indexes))
    sorted_indexes = np.sort(unique_indexes) # Sort from small to large
    eval_features = features[sorted_indexes]
    # index mapping from features to eval_features
    index_map = {}
    for new_i, old_i in enumerate(sorted_indexes):
      index_map[old_i] = new_i
    # encode cowatches by index_map
    eval_cowatches = []
    for old_cowatch in cowatches:
      eval_cowatch=[]
      for old_i in old_cowatch:
        eval_cowatch.append(index_map[old_i])
      eval_cowatches.append(eval_cowatch)
    return eval_features, eval_cowatches

  def mean_dist(self, embeddings, cowatches):
    """embeddings are l2 normalized.
    cosine_dist = a·b/|a||b| = a·b = Σ(x_a*x_b)
    embeddings: shape (num_embed, size_embed)
    cowatches: shape (num_cowatch, 2)
    """
    co_embeddings = embeddings[cowatches]  # shape (num_cowatch, 2, size_embed)
    distances = co_embeddings[:,0,:] * co_embeddings[:,1,:]
    distances = np.sum(distances, axis=-1)
    return np.mean(distances)




