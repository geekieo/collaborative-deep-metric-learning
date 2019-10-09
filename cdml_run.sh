#!/usr/bin/env bash
### 
# @Description: 服务全流程脚本
#    数据相关代码查看 ai-algorithm-cdml-data-process 项目。
# @Date: 2019-07-10 17:31:26
# @Author: Weng Jingyu
###
source /etc/profile
source /data/bigdata/env.sh

#本地工作路径
python_env=/data/anaconda2/envs/wengjy1/bin/python
week=`date +"%w"`
project_dir=/data/service/ai-algorithm-cdml
training_dir=$project_dir/training_dir
serving_dir=$project_dir/serving_dir
predict_dir=$serving_dir/predict_result
knn_dir=$serving_dir/knn_result

# HDFS (切换 HDFS 数据环境只需要修改 hdfs_dir)
training_hdfs_path=/user/prod/cdmldev/training_data
update_hdfs_path=/user/prod/cdmldev/update_data
knn_hdfs_path=/user/prod/cdmldev/knn_result
# 训练数据路径
training_click_records=$training_hdfs_path/click-records
training_dense_feature=$training_hdfs_path/dense-feature
training_feature_info=$training_hdfs_path/feature-info
# 更新数据路径
update_dense_feature=$update_hdfs_path/dense-feature
update_feature_info=$update_hdfs_path/feature-info
# 信号文件路径
training_signal=$training_hdfs_path/training.signal
update_signal=$update_hdfs_path/update.signal
refresh_signal=$knn_hdfs_path/refresh.signal

# 脚本日志
ip=`/sbin/ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
date=`date +"%Y%m%d"`
logfile=$project_dir/logs/cdml_run.log.$date

getDate(){ echo `date +"%Y-%m-%d|%H:%M:%S"`; }

check_training_task(){
    if [ $? -eq 0 ]; then
        printf "%s INFO $1 success.\n" $(getDate) >>$logfile
    else
        printf "%s WARNING $1 failed.\n" $(getDate) >>$logfile
        /usr/bin/curl -H "Content-Type: application/json" -X POST  --data '{"ars":"zhoukang@ifeng.com, wengjy1@ifeng.com","txt":"Training failed. Check log and serving_dir","sub":"CDML model service"}' http://rtd.ifeng.com/rotdam/mail/v0.0.1/send
        exit 0
    fi
}

check_update_task(){
    if [ $? -eq 0 ]; then
        printf "%s INFO $1 success.\n" $(getDate) >>$logfile
    else
        printf "%s WARNING $1 failed.\n" $(getDate) >>$logfile
        /usr/bin/curl -H "Content-Type: application/json" -X POST  --data '{"ars":"zhoukang@ifeng.com, wengjy1@ifeng.com","txt":"Update failed. Check log and serving_dir","sub":"CDML model service"}' http://rtd.ifeng.com/rotdam/mail/v0.0.1/send
        exit 0
    fi
}

check_train_timeout(){
    ps -ef | grep "$python_env online_data.py --base_save_dir" | grep -v grep | awk '{print $2}' | xargs kill
    ps -ef | grep "$python_env train.py --train_dir" | grep -v grep | awk '{print $2}' | xargs kill
}

check_update_timeout(){
    ps -ef | grep "$python_env predict.py --model_dir" | grep -v grep | awk '{print $2}' | xargs kill
    ps -ef | grep "$python_env faiss_knn.py --embedding_file" | grep -v grep | awk '{print $2}' | xargs kill
}

hadoop fs -test -e $training_signal
if [ $? -eq 0 ];then
    printf "%s INFO TRAIN:Start processing .\n" $(getDate) >$logfile
    cur_date=`date +"%Y%m%d%H"`
    check_train_timeout
    ## training
    hadoop fs -rm -r $training_signal
    check_training_task "TRAIN: delete training signal file"
    # hadoop fs 
    hadoop fs -getmerge $training_click_records $training_dir/dataset/click.records
    check_training_task "TRAIN: get training_dir/dataset/click.records"
    hadoop fs -getmerge $training_dense_feature $training_dir/dataset/dense.feature
    check_training_task "TRAIN: get training_dir/dataset/dense.feature"
    hadoop fs -getmerge $training_feature_info $training_dir/dataset/feature.info
    check_training_task "TRAIN: get training_dir/dataset/feature.info"

    # train model
    cd $project_dir
    $python_env online_data.py --base_save_dir $training_dir/dataset/ \
                               --training_click_records $training_dir/dataset/click.records \
                               --training_dense_feature $training_dir/dataset/dense.feature
    check_training_task "TRAIN: online_data"
    # 删除旧模型
    rm -rf $training_dir/checkpoints/*
    # 训练新模型
    $python_env train.py --train_dir $training_dir/dataset/cdml_1 \
                         --checkpoint_dir $training_dir/checkpoints
    check_training_task "TRAIN: train"
    # 部署
    # 模型 -> serving_dir
    mkdir -p $serving_dir/models/$cur_date
    cp -fr $training_dir/checkpoints/* $serving_dir/models/$cur_date
    # 重写 checkpoint 
    cd $serving_dir/models/$cur_date
    ckpt=`ls -lt | grep ckpt | head -1 |awk '{print $9}' |awk -F'.' '{print $2}'`
    printf "model_checkpoint_path: \"$serving_dir/models/$cur_date/model.$ckpt\"" > checkpoint
    # 部署完成信号
    trans_end_signal=$serving_dir/models/$cur_date/transend.signal
    touch $trans_end_signal
    check_training_task "TRAIN: copy ckpt -> serving_dir"

else
    hadoop fs -test -e $update_signal
    if [ $? -eq 0 ];then
        printf "%s INFO UPDATE:Start processing .\n" $(getDate) >$logfile
        cur_date=`date +"%Y%m%d%H"`             # 更新时间
        check_update_timeout
        ## updating
        hadoop fs -rm -r $update_signal
        check_update_task "UPDATE: delete update signal file"
        # hadoop fs 
        hadoop fs -getmerge $update_dense_feature $serving_dir/dataset/dense.feature
        check_update_task "UPDATE: get serving_dir/dataset/dense.feature"
        hadoop fs -getmerge $update_feature_info $serving_dir/dataset/feature.info
        check_update_task "UPDATE: get serving_dir/dataset/dense.feature"

        # serving_dir predict
        cd $project_dir 
        $python_env predict.py --model_dir $serving_dir/models  \
                               --feature_file $serving_dir/dataset/dense.feature \
                               --output_dir $predict_dir
        check_update_task "UPDATE: predict"
        # todo:calc knn result use guid_knn.py (input encoded features output knn)

        knn_result=$knn_dir/$cur_date        # 调整 knn_split 地址
        $python_env faiss_knn.py --embedding_file $predict_dir/output.npy \
                                 --decode_map_file $predict_dir/decode_map.json \
                                 --pred_feature_file $predict_dir/features.npy \
                                 --feature_info $serving_dir/dataset/feature.info
                                 --knn_result $knn_result
        check_update_task "UPDATE: faiss_knn"
        # put knn_result to hdfs and send signal
        target_dir=$knn_hdfs_path/$cur_date 
        hadoop fs -mkdir -p $target_dir
        hadoop fs -put -f $knn_result/knn_split* $target_dir
        check_update_task "UPDATE: knn_result -> hadoop"
        # set redis refresh signal 
        hadoop fs -touchz $refresh_signal
    else
        echo "nothing to do, waiting..."
        exit 1
    fi
fi

