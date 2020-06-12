# -*- coding:utf-8 -*-

"""

@Description
image data load and augmentation

@Reference
Zihao Dong, Ruixun Zhang, and Xiuli Shao. Scale-Recursive Network with Point Supervision for Crowd
Scene[J]. Neurocomputing, 2020, 384: 314-324.

"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import cv2


def data_augmentation(img, mp, count, load_size=720, fine_size=240, flip=True, is_test=False):
    """
    Process the original image and label. Training sets require data augmentation. 

    :param img: image sets
    :param mp: Density map 
    :param count: Point map
    :param load_size: image initial size 
    :param fine_size: image size after data augmentation
    :param flip: whether to flip the image, boolean variable
    :param is_test: whether it is test image, boolean variable
    :return: image and density map after data augmentation
    """
    if is_test:
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_tmp = img[h1:h1 + fine_size, w1:w1 + fine_size]
        mp_tmp = mp[h1:h1 + fine_size, w1:w1 + fine_size]
        count_tmp = count[h1:h1 + fine_size, w1:w1 + fine_size]
    else:
        iter_num = 0.0
        sum_mp = 0.0
        img_tmp = None
        mp_tmp = None
        count_tmp = None
        
        while (sum_mp < 10.0) and (iter_num < 10.0):
            h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            
            img_tmp = img[h1:h1 + fine_size, w1:w1 + fine_size]
            mp_tmp = mp[h1:h1 + fine_size, w1:w1 + fine_size]
            count_tmp = count[h1:h1 + fine_size, w1:w1 + fine_size]

            sum_mp = sum(sum(sum(mp_tmp))) / 3
            iter_num += 1

        # flip 
        if flip and np.random.random() > 0.5:
            img_tmp = np.fliplr(img_tmp)
            mp_tmp = np.fliplr(mp_tmp)
            count_tmp = np.fliplr(count_tmp)

    return img_tmp, mp_tmp, count_tmp


def load_data(image_path, args, flip=True, is_test=False, is_sample=False):
    """
    Load data operation 
    :param image_path: image set path
    :param args: configuration of global parameter 
    :param flip: whether to flip the image, boolean variable
    :param is_test: whether it is test image, boolean variable
    :param is_sample: whether it is validation image, boolean variable
    :return: result after the 
    """
    # 1. Get the corresponding path of train and test set
    if is_test:
        im_path = args.test_im_dir
        gt_path = args.test_gt_dir
        count_path = args.count_dir
    else:
        im_path = args.train_im_dir
        gt_path = args.train_gt_dir
        count_path = args.count_dir

    # 2. Get the full path of training data 
    name = image_path.split('/')[-1].split('.')[0]
    im_name = name + '.jpg' 
    gt_name = name + '.npy'     # Density map GT
    count_name = name + '.npy'  # Point map GT
    im_path += im_name
    gt_path += gt_name
    count_path += count_name

    # 3. Read the image, dentisy map and point map
    img = cv2.imread(im_path)
    mp = np.array(np.load(gt_path))
    count = np.array(np.load(count_path))
    if is_test:
        # Visualization of testing image and density map
        cv2.imwrite('./{}/om_{}.jpg'.format(args.test_dir, name), img)
        plt.imsave('./{}/og_{}.jpg'.format(args.test_dir, name), mp, cmap=plt.get_cmap('jet'))
        print("counting:%.2f" % (sum(sum(mp))))

    if is_sample:
        # Visualization of verification image and density map
        cv2.imwrite('./{}/om_{}.jpg'.format(args.sample_dir, name), img)
        plt.imsave('./{}/og_{}.jpg'.format(args.sample_dir, name), mp, cmap=plt.get_cmap('jet'))
        print("counting:%.2f" % (sum(sum(mp))))

    # 4. Density map from 1-channel to 3-channels
    mp = np.transpose(np.array([mp, mp, mp]), [1, 2, 0])
    count = np.transpose(np.array([count, count, count]), [1, 2, 0])
    
    # 5. Data augmentation with CROP operations 
    if is_test:
        img_tmp, mp_tmp,count_tmp = img, mp,count
    else:
        img_tmp, mp_tmp,count_tmp = data_augmentation(img, mp,count, load_size=args.load_size,
                                        fine_size=args.fine_size, flip=flip, is_test=is_test)

    # 6.  image and density map concatenate
    img_mp = np.concatenate((img_tmp, mp_tmp), axis=2)

    return img_mp, count_tmp


def get_real_count(gt_path, img_name):
    """
    Get the real crowd counting
    :param gt_path: Crowd density map path
    :param img_name: Crowd density map name
    """
    mp = np.array(np.load(gt_path))

    real_count = sum(sum(mp))
    map_name = "real_" + img_name
    print("Real count is %4d" % round(real_count))
    plt.imsave("../" + map_name + ".png", mp, cmap=plt.get_cmap('jet'))


# Test Script: Main
if __name__ == "__main__":
    img_path = "../data/data_gt/test_gt/"
    img_name = "IMG_1_B"
    get_real_count(img_path + img_name + ".npy", img_name)
