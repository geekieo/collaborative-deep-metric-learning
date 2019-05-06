# -*- encode:utf-8 -*-
import numpy as np
from parse_data import get_unique_watched_guids
from tensorflow import logging

logging.set_verbosity(logging.DEBUG)

def l2_normalize(a, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
  l2[l2==0] = 1
  return a / np.expand_dims(l2, axis)


class Evaluation():
  def __init__(self, features, cowatches):
    # NOTE features is very large
    try:
      self.features, self.cowatches = self._rencode(features, cowatches)
    except Exception as e:
      logging.warning('Evaluation.__init__ features or cowatches'+str(e))
      self.features, self.cowatches = None, None

  def _rencode(self, features, cowatches):
    unique_indexes = get_unique_watched_guids(cowatches)
    logging.info("Evaluater._rencode cowatches" + str(len(cowatches)) + 
        ". unique_indexes in cowatches: " + str(len(unique_indexes)))
    sorted_indexes = np.sort(unique_indexes) # Sort from small to large
    try:
      eval_features = features[sorted_indexes]
    except Exception as e:
      logging.error("Evaluater._rencode eval_features got None "+str(e))
      eval_features = None
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
    """ mean cosine distance of eval cowatch embeddings
    only work under the same set of hyperparameters
    embeddings are l2 normalized.
    cosine_dist = a·b/|a||b| = a·b = Σ(x_a*x_b)
    Arg:
      embeddings: shape (num_embed, size_embed)
      cowatches: shape (num_cowatch, 2)
    """
    try:
      co_embeddings = embeddings[cowatches]  # shape (num_cowatch, 2, size_embed)
    except Exception as e:
      logging.error("Evaluation.mean_dist "+str(e))
    distances = co_embeddings[:,0,:] * co_embeddings[:,1,:]
    distances = np.sum(distances, axis=-1)
    return np.mean(distances)

