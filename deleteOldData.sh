#!/usr/bin/env bash
### 
# @Description: 
 # @Date: 2019-07-10 17:31:26
 # @Author: Weng Jingyu
 ###
# 删除旧模型和旧结果
source /etc/profile
data=`date -d "7 days ago" +"%Y%m%d"`
rm -rf /data/service/ai-algorithm-cdml/serving_dir/knn_result/$data??
rm -rf /data/service/ai-algorithm-cdml/serving_dir/models/$data??
rm -rf /data/service/ai-algorithm-cdml/logs/update.$data.log
rm -rf /data/service/ai-algorithm-cdml/logs/training.$data.log