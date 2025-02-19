#!/bin/bash
source ~/.bashrc
path=$(dirname $0)
pwd
start=$(date +%s)

PYHON_HOME='/root/anaconda3/envs/tf2py37/bin'
dt=`date -d "1 day ago" +"%Y%m%d"`
aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/de_sample.dat data/sample.dat
aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/de_user.dat data/user.dat
aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/item.dat data/


cd main
$PYHON_HOME/python3 run_youtobednn.py
if [ $? -ne 0 ]; then
  echo "训练出错"
  exit -1
fi

/root/anaconda3/envs/tf1py36/bin/python sim.py


cd ../
tar -czvf models.tar.gz models
aws s3 cp models.tar.gz s3://transsion-algo-ind/offline/palmstore/sample/youtobednn_recall_v3/${dt}/

olddt=`date -d "3 day ago" +"%Y%m%d"`
aws s3 rm --recursive s3://transsion-algo-ind/offline/palmstore/sample/youtobednn_recall_v3/${olddt}/
end=$(date +%s)


# 计算运行时间  
runtime=$((end - start))

# 输出运行时间
echo "脚本运行时间：$runtime 秒"
#/root/anaconda3/envs/tf2py37/bin/python3 run_youtobednn.py
