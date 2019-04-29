import sys
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
from get_guid_title import get_video_coverimg_use_guid
from get_guid_title import get_video_name_use_guid
result_file = sys.stdout


nearestN=5

embedding_file='C:/Users/wengjy1/Desktop/VENet_190422_142926/output.npy'
decode_map_file='C:/Users/wengjy1/Desktop/decode_map.json'
encode_map_file='C:/Users/wengjy1/Desktop/encode_map.json'
query_guid_file = 'D:/PySpace/cdml/tests/guid.txt'


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

def calc_nn(query_index, all_embedding, decode_map):
  nearest_guids = []
  for index in query_index:
    dist = all_embedding.dot(all_embedding[index])
    closest_index = np.argsort(-dist, axis=0)[:nearestN]
    furthest_index = np.argsort(dist, axis=0)[0]
    result_file.write("\nnew : guid:---" + decode_map[index] + "--------\n")
    strp = "nearest "
    for i in range(nearestN):
      nearest_guids.append(decode_map[closest_index[i]])
      strp += decode_map[closest_index[i]] + ":" + str(
        all_embedding[closest_index[i]].dot(all_embedding[index])) + ","
    result_file.write(strp)
    result_file.write("\nfarthest i guid:---" + decode_map[furthest_index] + "--------{}".format(dist[furthest_index]))
  return nearest_guids


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


if __name__ == '__main__':
  np.random.seed(1234)
  ## calc knn
  decode_map = load_decode_map(decode_map_file)
  encode_map = load_encode_map(encode_map_file)
  EMBEDDINGS = load_embedding(embedding_file)
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
    print('\n%s#########################'%(title))   
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
    plt.savefig('D:/PySpace/cdml/results/{}.jpg'.format(index))
    # plt.show()
    plt.close()
