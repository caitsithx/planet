import glob
import os

import cv2

from settings import *

img_dirs = ["train-jpg", "test-jpg", "test-jpg-add"]

blacklist = ['Type_2/2845.jpg', 'Type_2/5892.jpg', 'Type_1/5893.jpg',
             'Type_1/1339.jpg', 'Type_1/3068.jpg', 'Type_2/7.jpg',
             'Type_1/746.jpg', 'Type_1/2030.jpg', 'Type_1/4065.jpg',
             'Type_1/4702.jpg', 'Type_1/4706.jpg', 'Type_2/1813.jpg',
             'Type_2/3086.jpg']
files_0522 = ['/Type_2/80.jpg', '/Type_3/968.jpg', '/Type_3/1120.jpg']


def try_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def resize_640(src_dir, tgt_dir, match_str):
    if not os.path.exists(src_dir):
        print("dir %s not exists" % src_dir)
        return

    try_mkdir(tgt_dir)

    files = glob.glob(src_dir + "/" + match_str)
    print("image num: %d" % len(files))
    for file_path in files:
        if file_path in blacklist:
            continue
            # print('')
            # print('skipping {}'.format(f))
        paths = file_path.split("/")
        file_name = paths[len(paths) - 1].split(".")[0]

        print('.', end='', flush=True)
        tgt_fn = tgt_dir + '/' + file_name

        if os.path.exists(tgt_fn):
            split = file_path.split('.')
            tgt_fn = tgt_dir + '/' + split[0] + '_add.' + split[1]
            # print(tgt_fn)

        img = cv2.imread(file_path)
        res = cv2.resize(img, (640, 640))
        cv2.imwrite(tgt_fn, res)


def resize_images():
    for img_dir in img_dirs:
        resize_640(DATA_DIR + "/" + img_dir,
                   DATA_DIR + "/" + img_dir + "-640", "*.jpg")


if __name__ == "__main__":
    print('creating resized images, this will take a while...')
    resize_images()
    print('done')
