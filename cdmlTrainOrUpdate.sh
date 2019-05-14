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
src_files=$project_dir/cdml_knn/$cur_date
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
    # todo: train model

    # todo:calc knn result use guid_knn.py (input encoded features output knn)

else
    hadoop fs -test -e $update_signal_file
    if [ $? -eq 0 ];then
        ## updating
        hadoop fs -rm -r $update_signal_file
        check_task "delete update signal file"
        # todo: calc knn results of new vectors

    else
        echo "nothing to do, waiting..."
        exit 1
    fi
fi

# put model to hdfs and send signal
hadoop fs -mkdir -p $target_dir
hadoop fs -put -f $src_files/knn_split* $target_dir

# set finish signal
hadoop fs -touchz $signal_file