#!/bin/bash
source ~/.bashrc
path=$(dirname $0)
pwd

PYHON_HOME='/root/anaconda3/envs/tf2py37/bin'
dt=`date -d "1 day ago" +"%Y%m%d"`

aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/sample.dat data/
aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/user.dat data/
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
aws s3 cp models.tar.gz s3://transsion-algo-ind/offline/palmstore/sample/youtobednn_recall/${dt}/


#/root/anaconda3/envs/tf2py37/bin/python3 run_youtobednn.py