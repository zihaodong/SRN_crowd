# -*- coding:utf-8 -*-
"""
@Function: Transform .mat to .npy

"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def mat2npy(mat_path, npy_path):
    """
    Transform .mat to .npy

    :param mat_path: Folder for .mat
    :param npy_path: Folder for .npy
    :return: None
    """
    mat_files = os.listdir(mat_path)
    for mat_file in mat_files:
        # load .mat
        mat_full_path = mat_path + '/' + mat_file
        train_gt = sio.loadmat(mat_full_path)
        train_gt = train_gt['outputD_map']

        # show
        # print(mat_file)
        # plt.imshow(train_gt)
        # plt.show()
        
        # output conversion information
        print (mat_file + " is transform successfully.")

        # save .npy
        mat_name = mat_file.split('.')[0]
        npy_full_path = npy_path + '/' + mat_name + '.npy'
        np.save(npy_full_path, train_gt)

if __name__ == '__main__':
    cur_path = sys.path[0]
    count_mat_path_ = '../data/part_B_final/train_data/train_count_gt'
    count_npy_path_ = '../data/train_count_gt'
    mat_path_ = '../data/part_B_final/train_data/train_gt'
    npy_path_ = '../data/train_gt'
    mat2npy(count_mat_path_, count_npy_path_)
    mat2npy(mat_path_, npy_path_)
