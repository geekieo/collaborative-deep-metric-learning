#!/usr/bin/env bash
# author @zhoukang

source /etc/profile
workdir=/data/prod/videoClickHistory
# fexist=hadoop fs -test -e /user/zhoukang/userknn/DSSM/signal.txt
day=`date +'%d%H'`
logfile=$workdir/log/log.log_$day

## check hdfs file existance
hadoop fs -test -e /user/zhoukang/videoknn/cdml/signal.txt
if [ $? -eq 1 ]
then
    echo 'no new model detected.'
    exit 0
fi

getDate(){ echo `date +"%Y-%m-%d|%H:%M:%S"`; }

check_task()
{
    if [ $? -eq 0 ]; then
        printf "%s INFO $1 success.\n" $(getDate) >>$logfile
    else
        printf "%s INFO $1 failed.\n" $(getDate) >>$logfile
        /usr/bin/curl -H "Content-Type: application/json" -X POST  --data '{"ars":"zhoukang@ifeng.com","txt":"cal video knn failed. check log on cluster","sub":"CDML KNN service"}' http://rtd.ifeng.com/rotdam/mail/v0.0.1/send
        exit 0
    fi
}
echo 'update model...' >$logfile
hadoop fs -rm -r /user/zhoukang/videoknn/cdml/signal.txt
check_task "delete signal file"

#echo 'update model...' >$logfile
## update hive table
cur_date=$(date  +'%Y%m%d%H')
knn_father_path=/user/zhoukang/videoknn/cdml
hadoop fs -test -d $knn_father_path/$cur_date/
if [ $? -eq 1 ]; then
    printf 'not exist: ',$knn_father_path/$cur_date/
    cur_date=$(date -d '1 hours ago'  +'%Y%m%d%H')
    hadoop fs -test -d $knn_father_path/$cur_date/
    if [ $? -eq 1 ];then
        printf 'not exist: ' + $knn_father_path/$cur_date/
        rm fdjakfdjkaslfjdasfjdasfjkdlasfjdklsajfkldajfkdfdafda
        check_task "check knn model"
    fi
fi
hive -e "set hive.exec.compress.output = false;
set hive.mapred.mode = nonstrict;
alter table recom.cdml_knn add partition (dt='$cur_date') location '/user/zhoukang/videoknn/cdml/$cur_date/';"
# check_task "update hive table"

cd $workdir/log
## get user history and put to hbase
hive -e "show partitions recom.cdml_knn" > partitions.txt
last=`cat partitions.txt | tail -1 | sed 's/dt=//g'`
i=0
d1=`date -d "$i days ago $last" +'%Y/%m/%d'`
d2=`date -d $d1 +'%Y%m%d'`

# ############## 新集群 ##################
cd $workdir
echo "[`date`] CDMLKnn use cdml_knn partition dt=$d2, last=$last" >>$logfile

/data/bigdata/core/spark/bin/spark-submit --class com.ifeng.recom.cdml.CDMLKnn  \
--master  yarn  \
--deploy-mode cluster \
--name KnnCDML \
--driver-memory 8g --driver-cores 1 \
--num-executors 50 --executor-memory 8g --executor-cores 1 \
--conf spark.default.parallelism=2000 \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.yarn.executor.memoryoverhead=2048 \
--conf spark.memory.fraction=0.8 \
--conf spark.memory.storageFraction=0.4 \
--conf spark.locality.wait=10s \
--conf spark.akka.frameSize=256  \
--conf spark.ui.retainedJobs=100 \
--conf spark.ui.retainedStages=100 \
--conf spark.worker.ui.retainedExecutors=100   \
--conf spark.worker.ui.retainedDrivers=100    \
--conf spark.akka.timeout=1000 \
--conf spark.task.maxFailures=30 \
--conf spark.port.maxRetries=100 \
--conf spark.network.timeout=10000s \
--conf spark.rpc.askTimeout=10000s \
--conf spark.speculation=true \
--conf spark.speculation.multiplier=10 \
--conf spark.kryo.registrator=com.ifeng.common.bean.MyRegistrator \
--conf "spark.executor.extraJavaOptions= -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -Xms8G -Xmn1G  -XX:MaxGCPauseMillis=50 -XX:GCPauseIntervalMillis=200  -XX:-UseAdaptiveSizePolicy -XX:-UseGCOverheadLimit  -XX:MaxTenuringThreshold=15 "  \
DataAlg-1.0-SNAPSHOT-shaded.jar dt=$last topk=30 >>$logfile
check_task "put filtered knn to redis"