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
    index = faiss.IndexHNSWFlat(factors, 60)  # M 越大，精准率增加，查询响应时间降低，索引时间增加，默认 32
    index.hnsw.efConstruction = 40  # efConstruction 越大，构建图的质量增加，搜索的精度增加，索引时间增加，默认 40
    index.hnsw.efSearch = 16        # efSearch 越大，精准率增加，查询的响应时间增加，默认 16
  elif method == 'L2':
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(factors) # L2 计算精准的索引
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)    # make it into a gpu index
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


def intersection(eI, fI):
  """取 eI 和 fI 的交集，对 eI 中存在交集的元素位置置 True
  Arg:
    eI, fI: int. 二维索引矩阵
  Return:
    mask: eI fI 每行元素集的交集在 eI 中的位置蒙版
  """
  mask = np.zeros(eI.shape).astype('bool')
  for i, (e, f) in enumerate(zip(eI, fI)):
    mask[i] = np.isin(e,f)
  return mask


def diff(eD, eI, fI):
  """对 eD 中 eI 和 fI 交集位置的元素置 0"""
  mask = intersection(eI, fI)
  eD[mask] = 0.0  # 会修改原始eD
  return eD


def iter_diff(eD, eI, fI, f_end=10):
  """对 eI 中每行每列元素，找出其在 fI 近邻和当前行元素集的交集，
     对 eI 中存在交集的元素位置,在 eD 对应位置元素置 0
  """
  begin = time.time()
  for e_row in eI:
    e_keep_row = []
    f_chk_row = [e_row[0]]
    for ei in e_row:
      if ei in f_chk_row:
        continue
      else:
        f_chk_row.append(fI[ei][1:f_end])
        f_chk_row = list(set(f_chk_row))
        e_keep_row.append(ei)
    mask_keep_row = np.isin(e_row,e_keep_row)
    mask_drop_row = (1-mask_keep_row).astype('bool')
    eD[i][mask_drop_row] = 0.0
  print('faiss_knn iter_diff cost time', time.time()-begin)
  return eD


def diff_progress(e_row, fI, eD, f_end):
  e_keep_row = []
  f_chk_row = [e_row[0]]
  for ei in e_row:
    if ei in f_chk_row:
      continue
    else:
      f_chk_row.append(fI[ei][1:f_end])
      e_keep_row.append(ei)
  mask_keep_row = np.isin(e_row,e_keep_row)
  mask_drop_row = (1-mask_keep_row).astype('bool')
  eD[i][mask_drop_row] = 0.0

def iter_diff_mp(eD, eI, fI, f_end=10):
  """对 eI 中每行每列元素，找出其在 fI 近邻和当前行元素集的交集，
     对 eI 中存在交集的元素位置,在 eD 对应位置元素置 0
  """
  begin = time.time()
  pool = Pool(processes=None, maxtasksperchild=6) # 使用最大进程
  for e_row in eI:
    result = pool.apply_async(diff_progress, args=(e_row, fI, eD))
  pool_my.close()
  pool_my.join()
  print('iter_diff_mp cost: %fs'%(end - begin))
  return eD


def calc_knn_desim(eD, eI, features, method='hnsw',nearest_num=51, desim_nearest_num=51):
  """
  Arg:
    eD, eI: 模型输出向量的 faiss search 结果
    features: 原始特征向量
    method: the same as the argument method in calc_knn
    nearest_num: the same as the argument nearest_num in calc_knn
    disim_gap: nearest_num of embeddings - nearest_num of features
  """
  print('calc features knn...')
  desim_nearest_num = desim_nearest_num if FLAGS.nearest_num > desim_nearest_num else FLAGS.nearest_num
  print('nearest_num:{}, desim_nearest_num:{}'.format(nearest_num, desim_nearest_num))
  _, fI = calc_knn(features, features, method, desim_nearest_num, l2_norm=True)
  np.save(FLAGS.topk_dir+'/fI.npy',fI)
  print('features knn done. fD.shape: ',eD.shape)
  # eD = diff(eD, eI, fI)
  eD = iter_diff_mp(eD, eI, fI)
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
  print('write_knn cost:%f s'%(end - begin))

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
