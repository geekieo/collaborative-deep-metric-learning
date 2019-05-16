# 切到 prod 用户
#!/usr/bin/env bash
# author @zhoukang
source /etc/profile
source /data/bigdata/env.sh

python_env=/data/anaconda2/bin/python2.7
week=`date +"%w"`
project_dir=/data/service/cdml
training_dir=$project_dir/src
serving_dir=$project_dir/src

cur_date=`date +"%Y%m%d%H"`
knn_files=$project_dir/knn_result/$cur_date        # 调整 knn_split 地址
target_dir=/user/zhoukang/videoknn/cdml/$cur_date 
# signal files
signal_file=/user/zhoukang/videoknn/cdml/signal.txt
update_signal_file=/user/zhoukang/video_clicks/cdml_update_signal.txt
training_signal_file=/user/zhoukang/video_clicks/cdml_training_signal.txt

hadoop fs -test -e $training_signal_file
if [ $? -eq 0 ];then
    ## training
    hadoop fs -rm -r $training_signal_file
    check_task "delete training signal file"
    # todo: hadoop fs 
    hadoop fs -getmerge /user/zhoukang/video_clicks/uid2records_cdml $project_dir/dataset/src_watch_history
    hadoop fs -getmerge /user/zhoukang/tables/cdml_video_vec $project_dir/dataset/src_features
    # todo: train model
    cd 
    /data/python online_data.py --base_save_dir  "tests/" --feature_file "tests/visual_features.txt" --watch_file "tests/watched_guids.txt"
    # 模型 -> serving_model

else
    hadoop fs -test -e $update_signal_file
    if [ $? -eq 0 ];then
        ## updating
        hadoop fs -rm -r $update_signal_file
        check_task "delete update signal file"
        # todo: hadoop fs 
        hadoop fs -getmerge /user/zhoukang/tables/cdml_video_vec $project_dir/dataset/src_features
        # serving_model predict 
        # todo:calc knn result use guid_knn.py (input encoded features output knn)
        # todo: calc knn results of new vectors
        # 先测 knn
    else
        echo "nothing to do, waiting..."
        exit 1
    fi
fi

# put model to hdfs and send signal
hadoop fs -mkdir -p $target_dir
hadoop fs -put -f $knn_files/knn_split* $target_dir

# set finish signal
hadoop fs -touchz $signal_file