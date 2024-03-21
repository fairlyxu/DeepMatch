#!/bin/bash
source ~/.bashrc
path=$(dirname $0)
pwd

PYHON_HOME='/root/anaconda3/envs/tf2py37/bin'
dt=$1

aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/sample.dat data/
aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/user.dat data/
aws s3 cp s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/item.dat data/


cd main
$PYHON_HOME/python3 main.py
if [ $? -ne 0 ]; then
  echo "训练出错"
  exit -1
fi

cd ../

tar -czvf models.tar.gz models
aws s3 cp models.tar.gz s3://transsion-algo-ind/offline/palmstore/sample/dssm_recall_v2/${dt}/
