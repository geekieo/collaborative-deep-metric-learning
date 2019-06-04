# -*- coding: utf-8 -*-
import sys
import os
import time
import faiss
from multiprocessing import Pool
import subprocess
import traceback
import numpy as np
import json
import tensorflow as tf
from tensorflow import flags

from utils import get_latest_folder


train_dir = "/data/wengjy1/cdml_1_unique"  # NOTE 路径是 data
ckpt_dir = train_dir+"/checkpoints/"
# ckpt_dir = get_latest_folder(checkpoints_dir, nst_latest=1)
embedding_file = ckpt_dir+'/output.npy'
decode_map_file = ckpt_dir+'/decode_map.json'
pred_feature_file = ckpt_dir+'features.npy'
topk_dir = ckpt_dir+'/knn_result'

FLAGS = flags.FLAGS
flags.DEFINE_string("embedding_file",embedding_file,
    "待计算近邻的向量文件")
flags.DEFINE_string("decode_map_file",decode_map_file,
    "向量文件索引到 guid 的映射文件")
flags.DEFINE_string("pred_feature_file",pred_feature_file,
    "原始特征向量文件")
flags.DEFINE_integer("nearest_num",51,
    "返回的近邻个数")
flags.DEFINE_string("topk_dir", ckpt_dir, 
    "Top-k 结果保存地址")


def load_embedding(filename):
  embeddings = np.load(filename)
  return embeddings


def load_decode_map(filename):
  decode_map = {}
  encode_map = {}
  with open(filename, 'r') as f:
    index2guid_str = json.load(f)
  for k,v in index2guid_str.items():
    decode_map[int(k)] = v
    encode_map[v] = int(k)
  return decode_map, encode_map


def calc_knn(embeddings, q_embeddings, method='hnsw',nearest_num=51, l2_norm=True):
  """use faiss to calculate knn for recall online
  Arg:
    embeddings
    nearest_num: 51 = 50 个近邻 + query 自身
    method: ['hnsw', 'L2', 'gpuivf'] supported
            'hnws': hnsw get high precision with low probes
            'L2': The only index that can guarantee exact results
            'gpuivf': inverted file index, low time cost with lossing little precision
    l2_norm: NOTE embeddings should be L2 normalized, otherwise it will get wrong result in HNSW method! 
  Return:
    D: D for distance. 对于 q_embeddings 中每个向量，在 embeddings 中的近邻索引
    I：I for index. 与 D 对应的每个近邻的与 query 向量的距离
  """
  if method not in ['hnsw', 'L2', 'gpuivf']:
    raise(ValueError, 'Unsupported method:%s'%(method))
  embeddings = embeddings.astype(np.float32)
  if l2_norm:
    norm_data = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norm_data
  factors = embeddings.shape[1]
  begin = time.time()
  single_gpu = True
  index = None
  if method == 'hnsw':
    index = faiss.IndexHNSWFlat(factors, nearest_num)  # M 越大，召回率增加，查询响应时间降低，索引时间增加，默认 32
    index.hnsw.efConstruction = 40  # efConstruction 越大，构建图的质量增加，搜索的精度增加，索引时间增加，默认 40
    index.hnsw.efSearch = 16        # efSearch 越大，召回率增加，查询的响应时间增加，默认 16
  elif method == 'L2':
    res = faiss.StandardGpuResources()
    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(factors)
    # make it into a gpu index
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)
  elif method == 'gpuivf':
    if single_gpu:
      res = faiss.StandardGpuResources()
      ## nlist=400, somewhat like cluster centroids, todo: search a good value pair (nlist, nprobes)!!!!
      index = faiss.GpuIndexIVFFlat(res, factors, 400, faiss.METRIC_INNER_PRODUCT)
    else:
      cpu_index = faiss.IndexFlat(factors)
      index = faiss.index_cpu_to_all_gpus(cpu_index)
    index.train(embeddings)

  index.add(embeddings)   # 待召回向量
  end = time.time()
  print('create index time cost:', end - begin)
  index.nprobe = 256      # 搜索聚类中心的个数
  D, I = index.search(q_embeddings, nearest_num)  # actual search , 全部缓存（训练+增量）作为query 
  end1 = time.time()
  print('whole set query time cost:', end1 - end)
  return D, I


def diff(eD, eI, fI):
  # 取 eI 和 fI 的交集，对 eD 中交集索引位置的元素置 0
  mask = np.zeros(eI.shape).astype('bool')
  for i, (e, f) in enumerate(zip(eI, fI)):
    mask[i] = np.isin(e,f)
  eD[mask] = 0.0  # 会修改原始eI
  return eD


def calc_knn_desim(eD, eI, features, method='hnsw',nearest_num=51, desim_gap=0):
  """
  Arg:
    eD, eI: 带召回向量的召回结果
    features: 原始特征向量
    method: the same as the argument method in calc_knn
    nearest_num: the same as the argument nearest_num in calc_knn
    disim_gap: nearest_num of embeddings - nearest_num of features
  """
  print('calc features knn...')
  desim_nearest_num = FLAGS.nearest_num-desim_gap if FLAGS.nearest_num > desim_gap else 1
  print('nearest_num:{}, desim_nearest_num:{}'.format(nearest_num, desim_nearest_num))
  fD, fI = calc_knn(features, features, method, desim_nearest_num, l2_norm=True)
  # np.save(FLAGS.topk_dir+'/fI.npy',fI)
  # np.save(FLAGS.topk_dir+'/fD.npy',fD)
  print('features knn done. fD.shape: ',eD.shape)
  eD = diff(eD, eI, fI)
  print('knn diff done. eD.shape: ',eD.shape)
  return  eD, eI


def write_process(path, index, begin_index, D, I):
  try:
    with open(os.path.join(path, 'knn_split'+str(index)), 'w') as fp:
      for i in range(I.shape[0]):
        query_id = DECODE_MAP[begin_index+i]
        nearest_ids = map(lambda x: DECODE_MAP[x], I[i][1:])
        nearest_scores = D[i][1:]
        ## larger than 0.5 for cosine and less than 1.0 for Euclidean distance under L2 norm
        ## 2(1-cosine) = e_dist
        topks = ''.join(map(
          lambda x: x[1][0] + "#" + str(x[1][1]) + '<' if  x[1][1] > 0.0 and x[1][1] <2.0 else "",
          enumerate(zip(nearest_ids, nearest_scores))))
        string = query_id + ',' + topks + '\n'
        fp.write(string)
  except Exception as e:
    print(traceback.format_exc())
    raise


def write_knn(topk_dir, split_num=10, D=None, I=None):
  if not os.path.exists(topk_dir):
    os.makedirs(topk_dir)
  total_num = D.shape[0]
  patch_num = total_num // split_num

  ## Pool method
  begin = time.time()
  pool_my = Pool(processes=split_num, maxtasksperchild=6)
  for i in range(split_num - 1):
    batch_begin = i * patch_num
    batch_end = (i + 1) * patch_num
    pool_my.apply_async(write_process, args=(topk_dir, i, batch_begin, D[batch_begin:batch_end], I[batch_begin:batch_end]))
  pool_my.apply_async(write_process, args=(topk_dir, split_num-1, (split_num-1)*patch_num,
                                         D[(split_num-1)*patch_num:], I[(split_num-1)*patch_num:]))
  pool_my.close()
  pool_my.join()
  end = time.time()
  print('multiprocessing pool:%f s'%(end - begin))

def main(args):
  # TODO logging FLAGS
  print("FLAGS.topk_dir " + str(FLAGS.topk_dir))
  print("FLAGS.decode_map_file " + str(decode_map_file))
  print("FLAGS.embedding_file " + str(embedding_file))

  subprocess.call('mkdir -p {}'.format(FLAGS.topk_dir), shell=True)
  global DECODE_MAP
  DECODE_MAP, _ = load_decode_map(FLAGS.decode_map_file)
  print("faiss_knn decode_map len", len(DECODE_MAP))
  embeddings = load_embedding(FLAGS.embedding_file)
  print("faiss_knn embedding_file shape", embeddings.shape)
  features = load_embedding(FLAGS.pred_feature_file)
  print("faiss_knn pred_feature_file shape", features.shape)
  print("calc_knn ...")
  D, I = calc_knn(embeddings, embeddings, method='hnsw', nearest_num=FLAGS.nearest_num)
  D, I = calc_knn_desim(D, I, features, method='hnsw',nearest_num=FLAGS.nearest_num)

  # np.save(FLAGS.topk_dir+'/D.npy',D)
  # np.save(FLAGS.topk_dir+'/I.npy',I)
  # D = np.load(FLAGS.topk_dir+'/D.npy')
  # I = np.load(FLAGS.topk_dir+'/I.npy')
    
  write_knn(topk_dir=FLAGS.topk_dir, split_num=10, D=D, I=I)
  print("faiss_knn knn_result have saved to FLAGS.topk_dir", FLAGS.topk_dir)

if __name__ == '__main__':
  tf.app.run()
