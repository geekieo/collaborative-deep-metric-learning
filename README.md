<!--
 * @Description: Readme
 * @Date: 2019-07-10 17:31:26
 * @Author: Weng Jingyu
 -->  

# Collaborative Deep Metric Learning for Video Recommendation

## 切到 v1.0 稳定版

```shell
git checkout v1.0
```

## requirement

|lib |command |
|-|-|
|python=3.7.3           |conda create -n wengjy1 python=3.7.3|
|cudatoolkit=9.0        |conda install cudatoolkit=9.0|
|tensorflow-gpu=1.13.1  |conda install tensorflow-gpu=1.13.1  |
|faiss-gpu=1.5.x        |conda install faiss-gpu cudatoolkit=9.0 -c pytorch |

### optional

opencv-python     4.1  
matplotlib  
pillow  
  
## Overview of Files

### Data

* `training_dir`
* `online_data.py`
* `parse_data.py`
* `imitation_data.py`
* ``

### Training

* `inputs.py`
* `models.py`
* `losses.py`
* `train.py`

### Evaluation

* `evaluate.py`

### Prediction

* `predict.py`
* `faiss_knn.py`

### Misc

* `utils.py`
* `get_guid_title.py`
* `show_knn.py`
* `README.md`

## 部署

### 切换 prod 用户

`su prod`

#### 没有 prod 用户则创建

`adduser prod`

### 下载项目

`cd /data/`  
`mkdir  -p /data/service`  
`git clone *项目git地址*`

### 修改路径权限

`chown -R prod:ifengdev cdml/`

### 添加定时任务

`crontab -e`  添加以下任务

```bash
# CDML Train or Update
*/5 * * * * /bin/bash /data/service/cdml/cdmlTrainOrUpdate.sh

# clean old models which generated 7 days ago
30 01 * * * /bin/bash /data/service/cdml/deleteOldData.sh
```
