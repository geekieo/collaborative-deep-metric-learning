import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from get_guid_title import get_video_coverimg_use_guid
from get_guid_title import get_video_name_use_guid
result_file = sys.stdout

INDEX2GUID = {}
EMBEDDINGS = None
nearestN=5

embedding_file='D:/Downloads/features.npy'
decode_map_file='C:/Users/wengjy1/Desktop/decode_map.json'


def load_embeddings(filename):
  global EMBEDDINGS
  EMBEDDINGS = np.load(filename)


def load_decode_map(filename):
  global INDEX2GUID
  with open(filename) as f:
    index2guid_str = json.load(f)
  for k,v in index2guid_str.items():
    INDEX2GUID[int(k)] = v


def calc_nn_user(userList, user_embedding):
  nearest_guids = []
  for rand_doc in userList:
    dist = user_embedding.dot(user_embedding[rand_doc][:, None])
    closest_doc = np.argsort(dist, axis=0)[-nearestN:][::-1]
    furthest_doc = np.argsort(dist, axis=0)[0][::-1]
    result_file.write("\nnew : guid:---" + INDEX2GUID[rand_doc] + "--------\n")
    strp = "nearest "
    for i in range(nearestN):
      nearest_guids.append(INDEX2GUID[closest_doc[i][0]])
      strp += INDEX2GUID[closest_doc[i][0]] + ":" + str(
        user_embedding[closest_doc[i][0]].dot(user_embedding[rand_doc])) + ","
    print(EMBEDDINGS[closest_doc[0][0]])
    print(np.sum(EMBEDDINGS[closest_doc[0][0]]-EMBEDDINGS[closest_doc[1][0]]))
    result_file.write(strp)
    result_file.write("\nfarthest i guid:---" + INDEX2GUID[furthest_doc[0]] + "--------{}".format(dist[furthest_doc][0][0]))
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
  return result_img


np.random.seed(1234)
## calc knn
load_decode_map(decode_map_file)
load_embeddings(embedding_file)

print(EMBEDDINGS.shape)

uni_EMBEDDINGS = np.unique(EMBEDDINGS, axis=0)
print(uni_EMBEDDINGS.shape)
print((EMBEDDINGS.shape[0]-uni_EMBEDDINGS.shape[0])/EMBEDDINGS.shape[0])  # Repetition rate

rand_doc = np.random.randint(0, EMBEDDINGS.shape[0], size=3)
for doc in rand_doc:
    # doc = 38418
    # doc = 547770
    # doc = 523234
    guids = calc_nn_user([doc], EMBEDDINGS)
    ## show results in image
    result_img = build_result([INDEX2GUID[doc]] + guids)
    print('\n%s#########################'%(get_video_name_use_guid(INDEX2GUID[doc])))
    for guid in guids:
        print(get_video_name_use_guid(guid))
    plt.imshow(result_img)
    plt.show()
