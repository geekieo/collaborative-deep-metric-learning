# -*- encode:utf-8 -*-
import numpy as np


def l2_normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def mean_dist(embeddings, cowatches):
  """embeddings are l2 normalized.
  cosine_dist = a·b/|a||b| = a·b = Σ(x_a*x_b)
  embeddings: shape (num_embed, size_embed)
  cowatches: shape (num_cowatch, 2)
  """
  co_embeddings = embeddings[cowatches]  # shape (num_cowatch, 2, size_embed)
  distances = co_embeddings[:,0,:] * co_embeddings[:,1,:]
  distances = np.sum(distances, axis=-1)
  return np.mean(distances)




