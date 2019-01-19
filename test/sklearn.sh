#!/bin/bash


# 0.19.1
# 0.19.2
# 0.20.0

DIR=$(dirname $0)
ENV=env3
PIP=$DIR/${ENV}/bin/pip
PYTHON=$DIR/${ENV}/bin/python
DENSE=$DIR/../train.dense

echo $PIP
echo $PYTHON

train_and_upload()
{   
    version=$1
    echo $version
    $PIP install scikit-learn==${version}
    $PYTHON ${DIR}/rf.py $DENSE $version
}

train_and_upload 0.18.1
train_and_upload 0.19.1
train_and_upload 0.19.2
train_and_upload 0.20.0

