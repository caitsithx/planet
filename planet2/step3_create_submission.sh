#!/bin/sh

python predict.py --ens $1
python predict.py --sub submit1.csv 0.98
