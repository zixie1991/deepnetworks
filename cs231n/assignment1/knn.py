#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np

cifar10_dir = './cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print 'Train data shape', X_train.shape
print 'Train labels shape', y_train.shape
print 'Test data shape', X_test.shape
print 'Test labels shape', y_test.shape

# visualize some samples from the dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
print num_classes
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
