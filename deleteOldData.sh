# 删除旧模型和旧结果
source /etc/profile
day=`date -d "7 days ago" +"%Y%m%d"`
rm -rf /data/service/ai-algorithm-cdml/serving_dir/knn_result/$day??
rm -rf /data/service/ai-algorithm-cdml/serving_dir/models/$day??