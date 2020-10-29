# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:03:49 2018

@author: 益慶
"""

import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")


# 加载数据
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    加载单批量的数据
    
    参数：
    cifar10_dataset_folder_path: 数据存储目录
    batch_id: 指定batch的编号
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    
    # features and labels
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

# 1.本地路径
cifar10_path = 'C:/Users/益慶/Downloads/cifar-10-batches-py' # 本地路径



# 共有5个batch的训练数据
x_train, y_train = load_cfar10_batch(cifar10_path, 1)
for i in range(2, 6):
    features, labels = load_cfar10_batch(cifar10_path, i)
    x_train, y_train = np.concatenate([x_train, features]), np.concatenate([y_train, labels])



from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

# 重塑
x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)

# 归一化
x_train = minmax.fit_transform(x_train_rows)
# 重新变为32 x 32 x 3
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
# 选出马的图像
images = x_train[y_train==1]

testImg=images[101]
for i in range(10,20):
    for j in range(10,20):
        testImg[i,j]=0