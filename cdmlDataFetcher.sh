#!/usr/bin/env bash
# author @zhoukang

source /etc/profile
if [ $# -ne 1 ];then
    echo "parameter 'isTraining' is need!"
    exit 1
fi

isTraining=0
if [ $1 -eq 1 ];then
    isTraining=1
fi

work_path=/data/prod/videoClickHistory
dayHour=`date +'%d%H'`
logfile=$work_path/log/log_dataFetcher.log_$dayHour

tsRecordFile=$work_path/tsRecord
minTS=`date -d $(date +"%Y%m%d") +"%s"`
test -e $tsRecordFile
if [ $? -eq 1 ]
then
    # file not exist
    printf $minTS > $tsRecordFile
fi
minTS=`tail -1 $tsRecordFile`
maxTS=`date +"%s"`

getDate(){ echo `date +"%Y-%m-%d|%H:%M:%S"`; }

check_task()
{
    if [ $? -eq 0 ]; then
        printf "%s INFO $1 success.\n" $(getDate) >>$logfile
    else
        printf "%s INFO $1 failed.\n" $(getDate) >>$logfile
        /usr/bin/curl -H "Content-Type: application/json" -X POST  --data '{"ars":"zhoukang@ifeng.com","txt":"cal video data failed. check log on cluster","sub":"CDML datafetcher service"}' http://rtd.ifeng.com/rotdam/mail/v0.0.1/send
        exit 0
    fi
}

if [ $isTraining -eq 1 ];then
    echo 'fetching training vectors...' >$logfile
else
    echo 'fetching new vectors...' >$logfile
fi

dayNum=7
startDate=`date -d "$dayNum days ago" +"%Y%m%d"`
endDate=`date +"%Y%m%d"`
today=`date +"%Y%m%d"`

## parameter 'vectorSaveHdfsPath' and 'userRecordsHdfsPath' should not being changed, unless you have an insight of the code!!!!
/data/bigdata/core/spark/bin/spark-submit --class com.ifeng.recom.cdml.cdmlProcess  \
--master  yarn  \
--deploy-mode cluster \
--name cdmlProcess \
--driver-memory 8g --driver-cores 1 \
--num-executors 80 --executor-memory 8g --executor-cores 1 \
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
DataAlg-1.0-SNAPSHOT-shaded.jar isTrain=$isTraining  startDate=$startDate endDate=$endDate minUserClickNum=10 maxUserClickNum=300 \
 minTS=$minTS maxTS=$maxTS minReadSec=10 today=$today vectorSaveHdfsPath=/user/zhoukang/tables/cdml_video_vec \
  userRecordsHdfsPath=/user/zhoukang/video_clicks/uid2records_cdml >>$logfile
check_task "fetching data success."

dataPath=/user/zhoukang/video_clicks

# send the signal
if [ $isTraining -eq 1 ];then
    hadoop fs -touchz $dataPath/cdml_training_signal.txt
    check_task "set training signal"
else
    hadoop fs -touchz $dataPath/cdml_update_signal.txt
    check_task "set update signal"
    printf $maxTS >$tsRecordFile
fi