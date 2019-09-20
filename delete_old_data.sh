#!/usr/bin/env bash
### 
# @Description: 
# @Date: 2019-07-10 17:31:26
# @Author: Weng Jingyu
###
# 删除旧结果
source /etc/profile
date=`date -d "7 days ago" +"%Y%m%d"`
rm -rf /data/service/ai-algorithm-cdml/serving_dir/knn_result/$date??
# 删除旧模型和日志
date=`date -d "30 days ago" +"%Y%m%d"`
rm -rf /data/service/ai-algorithm-cdml/serving_dir/models/$date??
rm -rf /data/service/ai-algorithm-cdml/logs/update.log.$date
rm -rf /data/service/ai-algorithm-cdml/logs/training.log.$date
rm -rf /data/service/ai-algorithm-cdml/logs/cdml_run.log.$date