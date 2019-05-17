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
checkpoints_dir = train_dir+"/checkpoints/"
ckpt_dir = get_latest_folder(checkpoints_dir, nst_latest=1)
embedding_file = ckpt_dir+'/output.npy'
decode_map_file = train_dir+'/decode_map.json'

FLAGS = flags.FLAGS
flags.DEFINE_string("embedding_file",embedding_file,
    "待计算近邻的向量文件")
flags.DEFINE_string("decode_map_file",decode_map_file,
    "待计算近邻的向量文件")
flags.DEFINE_integer("nearest_num",51,
    "返回的近邻个数")
flags.DEFINE_string("topk_path", None, 
    "Top-k 结果保存地址")
subprocess.call('mkdir -p {}'.format(FLAGS.topk_path), shell=True)

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


def knn_process(path, index, begin_index, I, D):
  try:
    with open(os.path.join(path, 'knn_split'+str(index)), 'w') as fp:
      for i in range(I.shape[0]):
        query_id = DECODE_MAP[begin_index+i]
        nearest_ids = map(lambda x: DECODE_MAP[x], I[i][1:])
        nearest_scores = D[i][1:]
        ##check knn distance metric, defualt:L2 norm(HNSW)!!!!  otherwise will get wrong result!!!!!
        ##recomment condition: larger than 0.5 for cosine and less than 1.0 for L2 norm
        ##TODO: filter similar video use simid!!!!!
        topks = '<'.join(map(
          lambda x: x[1][0] + "#" + str(x[1][1]) if x[1][1] < 1.0 else "",
          enumerate(zip(nearest_ids, nearest_scores))))
        string = query_id + ',' + topks + '\n'
        fp.write(string)
  except Exception as e:
    print(traceback.format_exc())
    raise


def calc_knn(embeddings, topk_path, nearest_num=51, split_num=10, D=None, I=None, method='hnsw',l2_norm=False):
  """use faiss to calculate knn for recall online
  Arg:
    embeddings
    nearest_num
    method: ['hnsw', 'L2', 'gpuivf'] supported
      'hnws' and 'L2' use cpu, hnsw get high precision with low probes
      'gpuivf': inverted file index, low time cost with lossing little precision
  """
  if method not in ['hnsw', 'L2', 'gpuivf']:
    raise(ValueError, 'Unsupported method:%s'%(method))
  if D is None and I is None:
    embeddings = embeddings.astype(np.float32)
    if l2_norm:
      norm_data = np.linalg.norm(embeddings, axis=1, keepdims=True)
      embeddings /= norm_data
    factors = embeddings.shape[1]
    begin = time.time()
    single_gpu = True
    index = None
    if method == 'hnsw':
      index = faiss.IndexHNSWFlat(factors, nearest_num)
      ## higher with better performance and more time cost
      index.hnsw.efConstruction = 40
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

    index.add(embeddings)
    end = time.time()
    print('create index time cost:', end - begin)
    index.nprobe = 256
    D, I = index.search(embeddings, nearest_num)  # actual search
    end1 = time.time()
    print('whole set query time cost:', end1 - end)
    # np.save(result_dir + '/nearest_index.npy', I)
    # np.save(result_dir + '/score.npy', D)
    # print("save to ",result_dir)

  total_num = len(DECODE_MAP)
  patch_num = total_num // split_num

  ## Pool method
  begin = time.time()
  pool_my = Pool(processes=split_num, maxtasksperchild=6)
  for i in range(split_num - 1):
    batch_begin = i * patch_num
    batch_end = (i + 1) * patch_num
    pool_my.apply_async(knn_process, args=(topk_path, i, batch_begin, I[batch_begin:batch_end], D[batch_begin:batch_end]))
  pool_my.apply_async(knn_process, args=(topk_path, split_num-1, (split_num-1)*patch_num,
                                         I[(split_num-1)*patch_num:], D[(split_num-1)*patch_num:]))
  pool_my.close()
  pool_my.join()
  end = time.time()
  print('multiprocessing pool:%f s'%(end - begin))

def main(args):
  global DECODE_MAP
  DECODE_MAP, _ = load_decode_map(FLAGS.decode_map_file)
  print("faiss_knn FLAGS.DECODE_MAP",FLAGS.decode_map_file, ' decode_map len', len(DECODE_MAP))
  embeddings = load_embedding(FLAGS.embedding_file)
  print("faiss_knn FLAGS.embedding_file",FLAGS.embedding_file, ' embedding_file shape', embeddings.shape)
  print("faiss_knn FLAGS.topk_path", FLAGS.topk_path, 'calc_knn ...')
  calc_knn(embeddings, topk_path=FLAGS.topk_path, nearest_num=FLAGS.nearest_num)

if __name__ == '__main__':
  tf.app.run()
