# Collaborative Deep Metric Learning for Video Understanding

## requirement

tensorflow-gpu    1.13.1  conda install tensorflow-gpu
faiss-gpu         1.5.1   conda install faiss-gpu cudatoolkit=9.0 -c pytorch  
  numpy           1.6.3

### optional

opencv-python     4.1
matplotlib
pillow
  
## Overview of Files

### Data

* `online_data.py`
* `parse_data.py`
* `imitation_data.py`
* ``

### Training

* `inputs.py`
* `models.py`
* `losses.py`
* `train.py`
* `utils.py`

### Evaluation

* `evaluate.py`
* `average_precision_calculator.py`（useless）
* `mean_average_precision_calculator.py`（useless）

### Prediction

* `predict.py`
* `faiss_knn.py`

### Misc

* `utils.py`
* `get_guid_title.py`
* `show_knn.py`
* `README.md`
* `流程表.md`

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
```
# CDML Train or Update
*/5 * * * * /bin/bash /data/service/cdml/cdmlTrainOrUpdate.sh

# clean old models which generated 7 days ago
30 01 * * * /bin/bash /data/service/cdml/deleteOldData.sh
```
