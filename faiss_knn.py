# -*- coding: utf-8 -*-
'''
@Description: 计算 knn 和结果去重
@Date: 2019-07-10 17:31:26
@Author: Weng Jingyu
'''
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

# 系统异步多线程并发线程数
os.environ['OMP_NUM_THREADS'] = '1'

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
flags.DEFINE_integer("desim_nearest_num",26,
    "原始特征向量近邻个数")
flags.DEFINE_string("topk_dir", ckpt_dir, 
    "Top-k 结果保存地址")


def load_embedding(filename):
  embeddings = np.load(filename)
  return embeddings


def load_decode_map(filename):
  """读取结果用 dict 返回"""
  decode_map = {}
  encode_map = {}
  with open(filename, 'r') as f:
    index2guid_str = json.load(f)
  for k,v in index2guid_str.items():
    decode_map[int(k)] = v
    encode_map[v] = int(k)
  return decode_map, encode_map


# def load_decode_map(filename):
#   """读取结果用 ndarray 返回
#   使用 ndarray 代替 dict 并未见到加速效果，且 60Mb 的 json 文件保存成 npy 文件将占据 172Mb，约为原始体积的 2.9 倍
#   """
#   with open(filename, 'r') as f:
#     index2guid_str = json.load(f)
#   decode_list = [None]*len(index2guid_str)
#   for k,v in index2guid_str.items():
#     decode_list[int(k)] = v
#   decode_map = np.asarray(decode_list)
#   return decode_map


def calc_knn(embeddings, q_embeddings, method='hnsw',nearest_num=51, l2_norm=True):
  """use faiss to calculate knn for recall online
  Arg:
    embeddings
    nearest_num: 51 = 50 个近邻 + query 自身
    method: 'hnws': hnsw get high precision with low probes
            'L2': The only index that can guarantee exact results
            'gpuivf': inverted file index, low time cost with lossing little precision
    l2_norm: NOTE embeddings should be L2 normalized, otherwise it will get wrong result in HNSW method! 
  Return:
    D: D for distance. 对于 q_embeddings 中每个向量，在 embeddings 中的近邻索引 
       向量在L2归一化后 euclid_dist = 2(1-cosine_dist)
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
    index = faiss.IndexHNSWFlat(factors, 55)  # M 越大，精准率增加，查询响应时间降低，索引时间增加，默认 32
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
  """对于 fD 超过 fD_threshold 的索引，fI 对应位置元素置-1
  这里认为 fD 超过 fD_threshold 的向量不相似"""
  # 使用 fD 过滤距离较远的 fI
  fI_drop_index = np.where(fD>fD_threshold)
  fI[fI_drop_index] = -1
  # 过滤自身
  for i, indexes in enumerate(fI):
    indexes[np.where(indexes==i)]=-1
  return fI

    
def add_invalid_row(eI, fI):
  """首行增加无效结果，0向量"""
  # eI fI 索引暂时 + 1，最后须 -1
  eI += 1
  fI += 1
  # fI 首行插入无效结果 0 向量
  zero_row = np.zeros((1, eI.shape[1]),dtype=np.int)
  eI = np.concatenate((zero_row, eI), axis=0)
  zero_row = np.zeros((1, fI.shape[1]),dtype=np.int)
  fI = np.concatenate((zero_row, fI), axis=0)
  return eI, fI


def get_next_fI(col_mask, col_eI, fI, f_end):
  """fI 首行已插入 0 向量"""
  col_eI = col_mask * col_eI
  return fI[col_eI][:,:f_end]


def desim_progress(progress_i, mask_dict, eI_patch, fI_patch):
  mask_patch = np.ones(eI_patch.shape).astype('bool') #默认True
  for i, (e, f) in enumerate(zip(eI_patch, fI_patch)):
    if e[0]==0:
      continue  # fI[0]==[0,0,0,0,...]
    else:
      mask_patch[i] =  (1 - np.isin(e,f))             #交集置False
  mask_dict[progress_i] = mask_patch


def iter_desim_mp(eI, fI, fD, fD_threshold=1.4, fI_end=31, process_num=None):
  begin = time.time()
  fI = fliter_fI(fI, fD, fD_threshold)
  eI, fI = add_invalid_row(eI, fI)
  keep_mask = np.ones(eI.shape).astype('bool')
  manager = mp.Manager()
  mask_dict = manager.dict()
  cpu_num = mp.cpu_count()
  process_num = cpu_num if process_num is None or process_num>cpu_num or process_num<=0 else process_num
  print('faiss_knn iter_desim_mp process_num: ',process_num)

  for col_i in range(eI.shape[1]):
    pool = mp.Pool(processes=process_num, maxtasksperchild=6)
    # print('col_i: ',col_i)
    col_keep_mask = keep_mask[:, col_i]
    col_eI = eI[:, col_i]
    next_fI = get_next_fI(col_keep_mask, col_eI, fI, fI_end)
    # print('next_fI[1:4]', next_fI[1:4])
    next_eI = eI[:, col_i:]
    # print('next_eI[1:4]', next_eI[1:4])
    patch_num = eI.shape[0]//process_num
    results = []
    for progress_i in range(process_num - 1):
      patch_begin = progress_i * patch_num
      patch_end = (progress_i + 1) * patch_num
      result = pool.apply_async(desim_progress, args=(progress_i, mask_dict, next_eI[patch_begin:patch_end],
                       next_fI[patch_begin:patch_end]))
      results.append(result)
    result = pool.apply_async(desim_progress, args=(process_num-1, mask_dict, next_eI[(process_num-1)*patch_num:],
                     next_fI[(process_num-1)*patch_num:]))

    results.append(result)
    # 等待所有进程函数执行完毕
    for result in results:
      result.wait() 
    pool.close()
    pool.join()

    # dict -> ndarray
    next_keep_mask=[]
    for i in range(process_num): # len(mask_dict) == process_num
      next_keep_mask.extend(mask_dict[i])
    next_keep_mask = np.asarray(next_keep_mask)
    # print('iter_desim_mp next_keep_mask',next_keep_mask[1:4],' shape ',next_keep_mask.shape)
    # 刷新 eI
    keep_mask[:,col_i:] = keep_mask[:,col_i:] * next_keep_mask
    # print('keep_mask', keep_mask[1:4])
    drop_mask = (1 - keep_mask[:, col_i:]).astype('bool')
    eI[:, col_i:][drop_mask] = 0
    # print('desimed eI', eI[1:4])

  # 过滤自身
  for i, indexes in enumerate(eI):
    indexes[np.where(indexes==i)] = 0
  # 恢复 eI ，去除首行无效0向量，同时索引-1
  eI = eI[1:] - 1
  print('faiss_knn iter_desim_mp cost: ',time.time()-begin)
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
  print('write_knn cost: %fs'%(end - begin))

# ============================ main ============================
def main(args):
  # TODO logging FLAGS
  global_begin=time.time()
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
  desim_nearest_num = FLAGS.desim_nearest_num if FLAGS.nearest_num > FLAGS.desim_nearest_num else FLAGS.nearest_num
  print('nearest_num:{}, desim_nearest_num:{}'.format(FLAGS.nearest_num, desim_nearest_num))
  fD, fI = calc_knn(features, features, method='hnsw', nearest_num=desim_nearest_num, l2_norm=True)
  np.save(FLAGS.topk_dir+'/fD.npy',fD)
  np.save(FLAGS.topk_dir+'/fI.npy',fI)

  # eD = np.load(FLAGS.topk_dir+'/eD.npy')
  # eI = np.load(FLAGS.topk_dir+'/eI.npy')
  # fD = np.load(FLAGS.topk_dir+'/fD.npy')
  # fI = np.load(FLAGS.topk_dir+'/fI.npy')
  # print('load eD eI fD fI')

  ## 去重
  # eI = desim(eI, fI)
  # np.save(FLAGS.topk_dir+'/eI_desim.npy',eI)
  eI = iter_desim_mp(eI, fI, fD)
  np.save(FLAGS.topk_dir+'/eI_iter_desim.npy',eI)
  
  global DECODE_MAP
  DECODE_MAP, _ = load_decode_map(FLAGS.decode_map_file)
  print("faiss_knn decode_map len", len(DECODE_MAP))
  # DECODE_MAP = load_decode_map(FLAGS.decode_map_file)
  # print("faiss_knn decode_map shape", DECODE_MAP.shape)

  # 备份 decode_map 至 topk_dir
  res = subprocess.Popen('cp %s %s'%(FLAGS.decode_map_file, FLAGS.topk_dir+'/decode_map.json'),
    shell=True,close_fds=True,bufsize=-1,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  # res = os.popen('cp %s %s'%(FLAGS.decode_map_file, FLAGS.topk_dir+'/decode_map.json'))
  print("faiss_knn backup decode_map by subprocess.Popen: ", res)
  res.wait()
  # np.save(FLAGS.topk_dir+'/decode_map.npy', DECODE_MAP)

  # 解析并保存最终结果
  write_knn(topk_dir=FLAGS.topk_dir, split_num=10, D=eD, I=eI)
  print("faiss_knn knn_result have saved to FLAGS.topk_dir", FLAGS.topk_dir)
  print("faiss_knn cost: %fs", time.time()-global_begin)

if __name__ == '__main__':
  tf.app.run()
