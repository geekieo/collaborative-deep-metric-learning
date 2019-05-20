# 切到 prod 用户
#!/usr/bin/env bash
# author @zhoukang
source /etc/profile
source /data/bigdata/env.sh

python_env=/data/anaconda2/envs/wengjy1/bin/python
week=`date +"%w"`
project_dir=/data/service/cdml
training_dir=$project_dir/training_dir
serving_dir=$project_dir/serving_dir
predict_dir=$serving_dir/knn_result

cur_date=`date +"%Y%m%d%H"`
# signal files
signal_file=/user/zhoukang/videoknn/cdml/signal.txt
update_signal_file=/user/zhoukang/video_clicks/cdml_update_signal.txt
training_signal_file=/user/zhoukang/video_clicks/cdml_training_signal.txt


dayHour=`date +'%d%H'`
logfile=$project_dir/log/log.log_$dayHour

getDate(){ echo `date +"%Y-%m-%d|%H:%M:%S"`; }

check_task()
{
    if [ $? -eq 0 ]; then
        printf "%s INFO $1 success.\n" $(getDate) >>$logfile
    else
        printf "%s INFO $1 failed.\n" $(getDate) >>$logfile
        /usr/bin/curl -H "Content-Type: application/json" -X POST  --data '{"ars":"zhoukang@ifeng.com, wengjy1@ifeng.com","txt":"training or update failed. check log on training server","sub":"CDML training service"}' http://rtd.ifeng.com/rotdam/mail/v0.0.1/send
        exit 0
    fi
}

printf "%s INFO Start processing .\n" $(getDate) >$logfile

hadoop fs -test -e $training_signal_file
if [ $? -eq 0 ];then

    ## training
    hadoop fs -rm -r $training_signal_file
    check_task "TRAIN: delete training signal file"
    # hadoop fs 
    hadoop fs -getmerge /user/zhoukang/video_clicks/uid2records_cdml $training_dir/dataset/watch_history
    check_task "TRAIN: get training_dir/dataset/watch_history"
    hadoop fs -getmerge /user/zhoukang/tables/cdml_video_vec $training_dir/dataset/features
    check_task "TRAIN: get training_dir/dataset/features"
    # train model
    cd $project_dir
    $python_env online_data.py --base_save_dir $training_dir/dataset/ \
                               --watch_feature_file $training_dir/dataset/features \
                               --watch_file $training_dir/dataset/watch_history
    check_task "TRAIN: online_data"
    # 删除旧模型
    rm -rf $training_dir/checkpoints/*
    # 训练新模型
    $python_env train.py --train_dir $training_dir/dataset/cdml_1_unique \
                         --checkpoint_dir $training_dir/checkpoints
    check_task "TRAIN: train"
    # TODO 新旧模型测试 看测试结果给部署信号，没有旧模型直接部署
    # 模型 -> serving_dir
    mkdir -p $serving_dir/models/$cur_date
    cp -fr $training_dir/checkpoints/* $serving_dir/models/$cur_date
    # 重写 checkpoint 
    cd $serving_dir/models/$cur_date
    ckpt=`ls -lt | grep ckpt | head -1 |awk '{print $9}' |awk -F'.' '{print $2}'`
    printf "model_checkpoint_path: \"$serving_dir/models/$cur_date/model.$ckpt\"" > checkpoint
    check_task "TRAIN: copy ckpt -> serving_dir"

else
    hadoop fs -test -e $update_signal_file
    if [ $? -eq 0 ];then
        ## updating
        hadoop fs -rm -r $update_signal_file
        check_task "UPDATE: delete update signal file"
        # hadoop fs 
        hadoop fs -getmerge /user/zhoukang/tables/cdml_video_vec $serving_dir/dataset/features
        check_task "UPDATE: get serving_dir/dataset/features"
        # wc -l $serving_dir/dataset/features > $serving_dir/dataset/features_line_num 

        # serving_dir predict
        cd $project_dir 
        $python_env predict.py --model_dir $serving_dir/models  \
                               --feature_file $serving_dir/dataset/features \
                               --output_dir $predict_dir
        check_task "UPDATE: predict"
        # todo:calc knn result use guid_knn.py (input encoded features output knn)
        cur_date=`date +"%Y%m%d%H"`             # 更新时间
        topk_path=$predict_dir/$cur_date        # 调整 knn_split 地址
        $python_env faiss_knn.py --embedding_file $predict_dir/output.npy \
                                 --decode_map_file $predict_dir/decode_map.json \
                                 --topk_path $topk_path
        check_task "UPDATE: faiss_knn"

        # put knn_result to hdfs and send signal
        target_dir=/user/zhoukang/videoknn/cdml/$cur_date 
        hadoop fs -mkdir -p $target_dir
        hadoop fs -put -f $topk_path/knn_split* $target_dir
        check_task "UPDATE: knn_result -> hadoop"
        # set finish signal
        hadoop fs -touchz $signal_file
    else
        echo "nothing to do, waiting..."
        exit 1
    fi
fi

