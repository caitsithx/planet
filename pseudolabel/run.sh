#!/bin/sh
echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

python train.py --train res50
python train.py --train dense161
if [ $? -ne '0' ]; then 
  exit 1
fi 

python train.py --train res101
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train dense201
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train dense121
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train inceptionv3
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train res152
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train dense169
if [ $? -ne '0' ]; then
  exit 1
fi
#python train.py --train res50
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train vgg19bn
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train vgg16bn

echo 'Training finished.'
