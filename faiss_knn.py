# -*- coding: utf-8 -*-
import sys
import os
import time
import faiss
import multiprocessing as mp
import subprocess
import traceback
import numpy as np
import json
import tensorflow as tf
from tensorflow import flags

from utils import get_latest_folder


serving_dir = "/data/wengjy1/cdml/serving_dir/"  # NOTE 路径是 data
ckpt_dir = serving_dir+"/predict_result/"
# ckpt_dir = get_latest_folder(checkpoints_dir, nst_latest=1)
embedding_file = ckpt_dir+'/output.npy'
decode_map_file = ckpt_dir+'/decode_map.json'
pred_feature_file = ckpt_dir+'features.npy'
topk_dir = serving_dir+'/knn_result'

FLAGS = flags.FLAGS
flags.DEFINE_string("embedding_file",embedding_file,
    "待计算近邻的向量文件")
flags.DEFINE_string("decode_map_file",decode_map_file,
    "向量文件索引到 guid 的映射文件")
flags.DEFINE_string("pred_feature_file",pred_feature_file,
    "原始特征向量文件")
flags.DEFINE_integer("nearest_num",81,
    "embedding 的近邻个数")
flags.DEFINE_integer("desim_nearest_num",41,
    "原始特征向量近邻个数")
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
    norm_data = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
    q_embeddings /= norm_data
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

# ============================ de-similarity  ============================
def desim(eI, fI):
  """取 eI 和 fI 的交集，对 eI 中存在交集的元素位置置 True
  Arg:
    eI, fI: int. 二维索引矩阵
  """
  drop_mask = np.zeros(eI.shape).astype('bool')
  for i, (e, f) in enumerate(zip(eI, fI)):
    drop_mask[i] = np.isin(e,f)
  eI[drop_mask] = -1
  return eI


def fliter_fI(fI, fD, fD_threshold):
  """对于 fD 超过 fD_threshold 的索引，fI对应位置元素置-1
  这里认为 fD 超过 fD_threshold 的向量不相似"""
  fI_drop_mask = np.zeros(fI.shape).astype('bool')
  fI_drop_index = np.where(fD>fD_threshold)
  fI_drop_mask[fI_drop_index] = True
  fI[fI_drop_mask] = -1
  return fI

    
def add_invalid_row(eI, fI):
  """首行增加无效结果，0向量"""
  # eI fI 索引暂时 + 1，最后访问 eD 时须 -1
  eI += 1
  fI += 1
  # fI 首行插入无效结果 0 向量
  zero_row = np.zeros((1, fI.shape[1]),dtype=np.int)
  eI = np.concatenate((zero_row, eI), axis=0)
  fI = np.concatenate((zero_row, fI), axis=0)
  return eI, fI


def get_next_fI(col_mask, col_eI, fI, f_end):
  """fI 首行已插入 0 向量"""
  col_eI = col_mask * col_eI
  return fI[col_eI][:,1:f_end]


def iter_desim(eI, fD, fI, fD_threshold=1.4, fI_end=30):
  """对 eI 中每行每列元素，找出其在 fI 近邻和当前行元素集的交集，
     对 eI 中存在交集的元素位置,在 eD 对应位置元素置 0
  Arg:
    eD,fD: 近邻距离。若向量经 L2 归一化，其值等于 2(1-余弦距离)。
    eI,fI: 近邻索引。
    fD_threshold： 若 f 经 L2 归一化，其范围为[0,2], 越小越接近。
    fI_end: fI 截取数量。
  """
  fI = fliter_fI(fI, fD, fD_threshold)
  eI, fI = add_invalid_row(eI, fI)
  
  keep_mask = np.ones(eI.shape).astype('bool')
  for col_i in range(eI.shape[1]):
    col_keep_mask = keep_mask[:, col_i]
    col_eI = eI[:, col_i]
    next_fI = get_next_fI(col_keep_mask, col_eI, fI, fI_end)
    next_eI = eI[:, col_i:]
    for i, (e, f) in enumerate(zip(next_eI, next_fI)):
      if e[0]==0:
        continue  # fI[0]==[0,0,0,0,...]
      else:
        keep_mask[:,col_i:][i] = (1 - np.isin(e,f)) * keep_mask[:,col_i:][i]
    iter_drop_mask = (1 - keep_mask[:, col_i:]).astype('bool')
    eI[:, col_i:][iter_drop_mask] = 0
    # print(col_i, eI[1])
    # print(col_i, keep_mask[1])
    # print(col_i, eI[2])
    # print(col_i, keep_mask[2])
    # print(col_i, eI[3])
    # print(col_i, keep_mask[3])
  eI[:,0] = -1 
  return eI


def desim_progress(progress_i, mask_dict, eI_patch, fI_patch):
  for i, (e, f) in enumerate(zip(eI_patch, fI_patch)):
    if e[0]==0:
      continue  # fI[0]==[0,0,0,0,...]
    else:
      mask_patch =  (1 - np.isin(e,f))
  mask_dict[progress_i] = mask_patch

def iter_desim_mp(eI, fD, fI, fD_threshold=1.4, fI_end=30, process_num=32):
  fI = fliter_fI(fI, fD, fD_threshold)
  eI, fI = add_invalid_row(eI, fI)
  keep_mask = np.ones(eI.shape).astype('bool')
  
  manager = mp.Manager()
  mask_dict = manager.dict()
  pool = mp.Pool(processes=process_num, maxtasksperchild=6)

  for col_i in range(eI.shape[1]):
    col_keep_mask = keep_mask[:, col_i]
    col_eI = eI[:, col_i]
    next_fI = get_next_fI(col_keep_mask, col_eI, fI, fI_end)
    next_eI = eI[:, col_i:]

    patch_num = eI.shape[0]//process_num
    for progress_i in range(process_num - 1):
      patch_begin = progress_i * patch_num
      patch_end = (progress_i + 1) * patch_num
      pool.apply(desim_progress, args=(progress_i, mask_dict, next_eI[patch_begin:patch_end],
                       next_fI[patch_begin:patch_end]))
    pool.apply(desim_progress, args=(process_num-1, mask_dict, next_eI[(process_num-1)*patch_num:],
                     next_fI[(process_num-1)*patch_num:]))
    # dict -> ndarray
    next_keep_mask=[]
    for i in range(process_num): # len(mask_dict) == process_num
      next_keep_mask.append(mask_dict[i])
    next_keep_mask = np.asarray(next_keep_mask)
    # 刷新 mask
    keep_mask[:,col_i:] = keep_mask[:,col_i:] * next_keep_mask
    drop_mask = (1 - keep_mask[:, col_i:]).astype('bool')
    eI[:, col_i:][drop_mask] = 0
    # print(col_i, eI[1])
    # print(col_i, keep_mask[1])
    # print(col_i, eI[2])
    # print(col_i, keep_mask[2])
    # print(col_i, eI[3])
    # print(col_i, keep_mask[3])
  pool.close()
  pool.join()
  eI[:,0] = -1 
  return eI


# ============================ write result ============================
def write_by_D_process(path, index, begin_index, D, I):
  """丢弃ID在D中值为 0.0"""
  try:
    with open(os.path.join(path, 'knn_split'+str(index)), 'w') as fp:
      for i in range(I.shape[0]):
        query_id = DECODE_MAP[begin_index+i]
        nearest_ids = map(lambda x: DECODE_MAP[x], I[i][1:])
        nearest_scores = D[i][1:]
        ## larger than 0.5 for cosine and less than 1.0 for Euclidean distance under L2 norm
        ## 2(1-cosine) = e_dist
        topks = ''.join(map(
          lambda x: x[1][0] + "#" + str(x[1][1]) + '<' if  x[1][1] > 0.0 and x[1][1] <1.4 else "",
          enumerate(zip(nearest_ids, nearest_scores))))
        string = query_id + ',' + topks + '\n'
        fp.write(string)
  except Exception as e:
    print(traceback.format_exc())
    raise


def write_process(path, index, begin_index, D, I):
  """丢弃ID在I中值为-1"""
  try:
    with open(os.path.join(path, 'knn_split'+str(index)), 'w') as fp:
      for i in range(I.shape[0]):
        query_id = DECODE_MAP[begin_index+i]
        nearest_ids = I[i][1:]
        nearest_scores = D[i][1:]
        topks = ''.join(map(
          lambda x: DECODE_MAP[x[1][0]] + "#" + str(x[1][1]) + '<' if  x[1][0] > 0 and x[1][1] > 0.0 and x[1][1] <1.4 else "",
          enumerate(zip(nearest_ids, nearest_scores)))) # x[1][0]为近邻索引, x[1][1]为近邻距离
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
  pool_my = mp.Pool(processes=split_num, maxtasksperchild=6)
  for i in range(split_num - 1):
    patch_begin = i * patch_num
    patch_end = (i + 1) * patch_num
    pool_my.apply_async(write_process, args=(topk_dir, i, patch_begin, D[patch_begin:patch_end], I[patch_begin:patch_end]))
  pool_my.apply_async(write_process, args=(topk_dir, split_num-1, (split_num-1)*patch_num,
                                         D[(split_num-1)*patch_num:], I[(split_num-1)*patch_num:]))
  pool_my.close()
  pool_my.join()
  end = time.time()
  print('write_knn cost:%f s'%(end - begin))


# ============================ main ============================
def main(args):
  # TODO logging FLAGS
  print("FLAGS.topk_dir " + str(FLAGS.topk_dir))
  print("FLAGS.decode_map_file " + str(decode_map_file))
  print("FLAGS.embedding_file " + str(embedding_file))

  subprocess.call('mkdir -p {}'.format(FLAGS.topk_dir), shell=True)

  embeddings = load_embedding(FLAGS.embedding_file)
  print("faiss_knn embedding_file shape", embeddings.shape)
  features = load_embedding(FLAGS.pred_feature_file)
  print("faiss_knn pred_feature_file shape", features.shape)

  print("faiss_knn calc_knn embeddings...")
  eD, eI = calc_knn(embeddings, embeddings, method='hnsw', nearest_num=FLAGS.nearest_num)
  np.save(FLAGS.topk_dir+'/eD.npy',eD)
  np.save(FLAGS.topk_dir+'/eI.npy',eI)

  print('faiss_knn calc_knn features...')
  desim_nearest_num = desim_nearest_num if FLAGS.nearest_num > desim_nearest_num else FLAGS.nearest_num
  print('nearest_num:{}, desim_nearest_num:{}'.format(nearest_num, desim_nearest_num))
  fD, fI = calc_knn(features, features, method='hnsw', nearest_num=FLAGS.desim_nearest_num, l2_norm=True)
  np.save(FLAGS.topk_dir+'/fD.npy',fD)
  np.save(FLAGS.topk_dir+'/fI.npy',fI)

  # eD = np.load(FLAGS.topk_dir+'/eD.npy')
  # eI = np.load(FLAGS.topk_dir+'/eI.npy')
  # fD = np.load(FLAGS.topk_dir+'/fD.npy')
  # fI = np.load(FLAGS.topk_dir+'/fI.npy')

  ## 去重
  # eI = desim(eI, fI)
  # eI = iter_desim(eI, fI, fD)
  eI = iter_desim_mp(eI, fI, fD)
  np.save(FLAGS.topk_dir+'/eI_desim.npy',eI)
  
  global DECODE_MAP
  DECODE_MAP, _ = load_decode_map(FLAGS.decode_map_file)
  print("faiss_knn decode_map len", len(DECODE_MAP))
  # 保留 decode_map
  with open(FLAGS.topk_dir+'/decode_map.json', 'w') as file:
    json.dump(DECODE_MAP, file, ensure_ascii=False)

  # 解析并保存最终结果
  write_knn(topk_dir=FLAGS.topk_dir, split_num=10, D=eD, I=eI)
  print("faiss_knn knn_result have saved to FLAGS.topk_dir", FLAGS.topk_dir)

if __name__ == '__main__':
  tf.app.run()
