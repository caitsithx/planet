## OS and Hardware requirement
This model was developed on following software and hardware platform:
+ GTX 1080ti GPU with 11GB GPU Memory
+ Ubuntu 16.04
+ CUDA 8.0
+ cuDNN 5.1
+ Anaconda 3 and Python 3.6 with following packages:
  - cv2 (conda install opencv)
  - pytorch,torchvision (conda install pytorch torchvision cuda80 -c soumith)
  - bcolz (conda install bcolz)

This model has not been tested on other GPU or OS.

## Steps to train and create submission file:
1. Prepare the data, download all the 7z files put them into a single directory, let's call this directory as INPUT_PATH. Create an subdirectory "additional" under INPUT_PATH and move the additional_Type_1_v2.7z, additional_Type_2_v2.7z, additional_Type_3_v2.7z files into INPUT_PATH/additional directory. Then unpack all the 7z files.
Now under your INPUT_PATH you should have "train", "test" and "additional" subdirectories. 
```
chicm@chicm-ubuntu:~/ml/kgdata/input$ ls
additional  test  train
```
Under "additional" directory you should have "Type_1", "Type_2" and "Type_3" subdirectories.
```
chicm@chicm-ubuntu:~/ml/kgdata/input/additional$ ls
Type_1  Type_2  Type_3
```
2. Open settings.py, change the value of INPUT_PATH to the absolute path of your INPUT_PATH directory, without a slash '/' at the end.

3. Enter the directory where the step1_preprocess_images.sh is located, Run ./step1_preprocess_images.sh, This step normally take around half an hour on my computer.
```
./step1_preprocess_images.sh
```
Please note, the above script must be executed from the directory where it is located. 

4. Run ./step2_train_models.sh, This step normally take around 24 hours on my GTX1080ti GPU.
```
./step2_train_models.sh
```
5. To generate submission file for stage 1 test data, run ./step3_create_submission.sh, Then you will get stage 1 test data submission file submit1.csv in INPUT_PATH/resize640/results directory.
```
./step3_create_submission.sh stage1
```
To generate submission file for stage 2 test data, firstly open settings.py, set the value of STATGE2_TEST_DATA_PATH to the absolute path of the directory which contains stage 2 test pictures, then run following script:
```
./step3_create_submission.sh stage2
```

