import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from skimage import io, transform

from settings import *

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

    file_num = len(files)
    print("image num: %d" % file_num)
    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for file_path in files:
            if file_path in blacklist:
                continue
                # print('')
                # print('skipping {}'.format(f))
            paths = file_path.split("/")
            file_name = paths[len(paths) - 1]

            tgt_fn = tgt_dir + '/' + file_name

            if os.path.exists(tgt_fn):
                split = file_path.split('.')
                tgt_fn = tgt_dir + '/' + split[0] + '_add.' + split[1]

            futures.append(executor.submit(resize_image_640, file_path,
                                           tgt_fn))
            # resize_image_640(file_path, tgt_fn)

        for f in as_completed(futures):
            file_num -= 1
            if file_num % 100 == 0:
                print("remaining %d images to resize" % file_num)


def resize_image_640(src_img_path, tgt_img_path):
    # print("resize %s to %s" % (src_img_path, tgt_img_path))
    img = io.imread(src_img_path)
    # print(img.dtype)
    res = transform.resize(img, (640, 640), preserve_range=True,
                           mode='constant')
    # print(res.dtype)
    io.imsave(tgt_img_path, res.astype(np.uint8))


def resize_images():
    for img_dir in img_dirs:
        resize_640(DATA_DIR + "/" + img_dir,
                   DATA_DIR + "/" + img_dir + "-640", "*.jpg")


if __name__ == "__main__":
    print('creating resized images, this will take a while...')
    resize_images()
    print('done')
