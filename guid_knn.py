# -*- coding: utf-8 -*-
import sys
import os
import cv2
import time
import faiss
from multiprocessing import Pool
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
from utils import get_latest_folder
from get_guid_title import get_video_coverimg_use_guid
from get_guid_title import get_video_name_use_guid


nearestN=5

train_dir = "/data/wengjy1/cdml_2"  # NOTE 路径是 data
checkpoints_dir = train_dir+"/checkpoints/"
ckpt_dir = get_latest_folder(checkpoints_dir,nst_latest=1)
embedding_file = ckpt_dir+'/output.npy'
decode_map_file = train_dir+'/decode_map.json'
encode_map_file = train_dir+'/encode_map.json'
# 召回文件地址
knn_result_path = './cdml_knn/'
# 测试文件地址
query_guid_file = '/home/wengjy1/cdml/tests/guid.txt'
result_file = ckpt_dir+'/results'


def load_embedding(filename):
  EMBEDDINGS = np.load(filename)
  return EMBEDDINGS


def load_decode_map(filename):
  decode_map ={}
  with open(filename, 'r') as f:
    index2guid_str = json.load(f)
  for k,v in index2guid_str.items():
    decode_map[int(k)] = v
  return decode_map

def load_encode_map(filename):
  encode_map = {}
  with open(filename, 'r') as f:
    guid2index_str = json.load(f)
  for k,v in guid2index_str.items():
    encode_map[k] = int(v)
  return encode_map


def load_guids(filename):
  guids=[]
  with open(filename, 'r') as file:
    for line in file.readlines():
      line = line.strip()
      guids.append(line)
  return guids


def calc_nn(query_index, embeddings, decode_map):
  """use numpy to calculate knn for test show"""
  nearest_guids = []
  for index in query_index:
    dist = embeddings.dot(embeddings[index])
    closest_index = np.argsort(-dist, axis=0)[:nearestN]
    furthest_index = np.argsort(dist, axis=0)[0]
    print("\nnew : guid:---" + decode_map[index] + "--------\n")
    strp = "nearest "
    for i in range(nearestN):
      nearest_guids.append(decode_map[closest_index[i]])
      strp += decode_map[closest_index[i]] + ":" + str(
        embeddings[closest_index[i]].dot(embeddings[index])) + ","
    print(strp)
    print("\nfarthest i guid:---" + decode_map[furthest_index] + "--------{}".format(dist[furthest_index]))
  return nearest_guids


def knn_process(path, index, deocde_map, begin_index, I, D):
  try:
    with open(os.path.join(path, 'knn_split'+str(index)), 'w') as fp:
      for i in range(I.shape[0]):
        query_id = deocde_map[begin_index+i]
        nearest_ids = map(lambda x: deocde_map[x], I[i][1:])
        nearest_scores = D[i][1:]
        topks = '<'.join(map(
          lambda x: x[1][0] + "#" + str(x[1][1]) if x[1][1] < 1.0 else "",
          enumerate(zip(nearest_ids, nearest_scores))))
        string = query_id + ',' + topks + '\n'
        fp.write(string)
  except Exception as e:
    print(str(e))
    raise


def calc_knn(embeddings, nearest_num=51, D=None, I=None, method='hnsw',l2_norm=False):
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
        np.save(knn_result_path + 'nearest_index.npy', I)
        np.save(knn_result_path + 'score.npy', D)

    total_num = len(guid2id_dict)
    splitted_num = total_num // split_num

    ## FIXME: Pool method
    begin = time.time()
    pool_my = Pool(processes=split_num, maxtasksperchild=6)
    for i in range(split_num - 1):
        batch_begin = i * splitted_num
        batch_end = (i + 1) * splitted_num
        pool_my.apply_async(knn_process, args=(topk_path, i, batch_begin, I[batch_begin:batch_end], D[batch_begin:batch_end]))
    pool_my.apply_async(knn_process, args=(topk_path, split_num-1, (split_num-1)*splitted_num,
                                           I[(split_num-1)*splitted_num:], D[(split_num-1)*splitted_num:]))
    pool_my.close()
    pool_my.join()
    end = time.time()
    print('multiprocessing pool:%f s'%(end - begin))


def build_result(guids):
  vertical_grid = 20
  horizontal_grid = 10
  half_index = int(np.ceil(len(guids) / 2))
  h_len = len(guids[:half_index])
  result_w = 640
  result_h = 480
  result_img = np.ones((result_h + vertical_grid, result_w + h_len * horizontal_grid,3), np.uint8) * 255
  grid_w = int(np.ceil(result_w / h_len))
  grid_h = result_h // 2
  for i, guid in enumerate(guids[:half_index]):
    img = get_video_coverimg_use_guid(guid)
    if img is not None:
      re_img = cv2.resize(img, dsize=(grid_w,grid_h))
      re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)
      if re_img.ndim == 2:
        re_img = np.expand_dims(re_img, axis=2)
        re_img = np.repeat(re_img, repeats=3, axis=2)
      result_img[:grid_h, i * (horizontal_grid + grid_w): i * horizontal_grid + (i+1) * grid_w,:] = re_img
  for i, guid in enumerate(guids[half_index:]):
    img = get_video_coverimg_use_guid(guid)
    if img is not None:
      re_img = cv2.resize(img, dsize=(grid_w, grid_h))
      re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)
      if re_img.ndim == 2:
        re_img = np.expand_dims(re_img, axis=2)
        re_img = np.repeat(re_img, repeats=3, axis=2)
      result_img[grid_h+vertical_grid:grid_h*2+vertical_grid, i * (horizontal_grid + grid_w): i * horizontal_grid + (i + 1) * grid_w,:] = re_img
  return  result_img


def set_matplot_zh_font():
  myfont = mpl.font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
  mpl.rcParams['axes.unicode_minus'] = False
  mpl.rcParams['font.sans-serif'] = ['SimHei']


def show_test():
  if not os.path.exists(result_file):
    os.mkdir(result_file)
  # print 写入文件
  f_handler=open('guid_knn_pc_test.log', 'w')
  sys.stdout=f_handler
  ## calc knn
  decode_map = load_decode_map(decode_map_file)
  encode_map = load_encode_map(encode_map_file)
  EMBEDDINGS = load_embedding(embedding_file)
  # np.random.seed(1234)
  # query_ids = np.random.randint(0, EMBEDDINGS.shape[0], size=100)
  query_guids = load_guids(query_guid_file)

  print(EMBEDDINGS.shape)
  uni_EMBEDDINGS = np.unique(EMBEDDINGS, axis=0)
  print(uni_EMBEDDINGS.shape)
  print((EMBEDDINGS.shape[0]-uni_EMBEDDINGS.shape[0])/EMBEDDINGS.shape[0])  # Repetition rate

  set_matplot_zh_font()
  for guid in query_guids:
    index = encode_map[guid]
    nearest_guids = calc_nn([index], EMBEDDINGS, decode_map)
    ## show results in image
    result_img = build_result([decode_map[index]] + nearest_guids)
    titles = []
    title = get_video_name_use_guid(guid)
    print('\n%s###Query###'%(title))
    titles.append(title)
    for guid in nearest_guids:
        title = get_video_name_use_guid(guid)
        print(title)
        titles.append(title)
    ret = np.ones((result_img.shape[0] * 3 // 2, result_img.shape[1], result_img.shape[2]), dtype=np.uint8) * 255
    
    ret[-result_img.shape[0]:, :, :] = result_img
    plt.imshow(ret)
    plt.text(320, 10, '\n'.join(titles), fontsize=12, ha='center', va='top')
    plt.title('标题顺序--从左至右从上至下')
    
    plt.axis('off')
    plt.savefig(result_file+'/{}.jpg'.format(index))
    # plt.show()
    plt.close()


def knn_test():
  embeddings = load_embedding(embedding_file)
  calc_knn(embeddings, nearest_num=41)

if __name__ == '__main__':
  pass
