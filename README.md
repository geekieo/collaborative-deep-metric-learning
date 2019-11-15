<!--
 * @Description: Readme
 * @Date: 2019-07-10 17:31:26
 * @Author: Weng Jingyu
 -->  

# Collaborative Deep Metric Learning for Video Recommendation

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

## CDML模型工程部署

### 线上服务

服务器：10.80.98.152
工程路径： /data/service/ai-algorithm-cdml

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

## CDML数据工程部署

### 数据工程项目地址

项目地址：https://git.ifengidc.com/ai/algorithm/ai-algorithm-common-data-process-op
样本获取代码：src/main/scala/com/ifeng/recom/cdml/cdmlProcess.scala
结果上传代码：src/main/scala/com/ifeng/recom/cdml/CDMLKnn.scala

### 线上服务

服务器：10.80.17.148
工程路径：/data/prod/videoClickHistory

### 切换 prod 用户

`su prod`

### 上传项目文件

工程打成jar包，上传 spark 服务器所在工程路径
上传工程路径 src/main/scala/com/ifeng/shell/ 下的 cdmlDataFetcher.sh 和 cdml_knn.sh

### 添加定时任务

```bash
# fetch cdml training data, 1 a.m. and 13 p.m.
0 01,13 * * * /bin/bash /data/prod/videoClickHistory/cdmlDataFetcher.sh 1

# fetch cdml update data, every 2 hours
40 */2 * * * /bin/bash /data/prod/videoClickHistory/cdmlDataFetcher.sh 0

# update cdml knn result to redis
*/5 * * * * /bin/bash /data/prod/videoClickHistory/cdml_knn.sh
```
