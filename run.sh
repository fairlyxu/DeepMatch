#!/bin/bash
source ~/.bashrc
path=$(dirname $0)
pwd

PYHON_HOME='/root/anaconda3/envs/tf2py37/bin/'

dt=`date -d "1 day ago" +"%Y%m%d"`
aws s3 cp s3://transsion-algo-ire/offline/palmstore/sample/dssm_recall_v2/${dt}/sample.dat data/
aws s3 cp s3://transsion-algo-ire/offline/palmstore/sample/dssm_recall_v2/${dt}/user.dat data/
aws s3 cp s3://transsion-algo-ire/offline/palmstore/sample/dssm_recall_v2/${dt}/item.dat data/


cd main
$PYHON_HOME/python3 main.py
if [ $? -ne 0 ]; then
  echo "训练出错"
  exit -1
fi

cd ../
zip  models.zip models
aws s3 cp models.zip s3://transsion-algo-ire/offline/palmstore/sample/dssm_recall_v2/${dt}/models.zip


