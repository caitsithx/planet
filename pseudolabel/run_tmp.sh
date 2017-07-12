#python train.py --train dense169
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train vgg16bn
if [ $? -ne '0' ]; then
  exit 1
fi
python train.py --train vgg19bn
if [ $? -ne '0' ]; then
  exit 1
fi

