from mnist import MNIST
import scipy.io
import numpy as np
import random
from scipy.misc import imresize
from skimage.color import rgb2gray

random.seed(12345)

def get_fashion_mnist(num_examples, unlabeled=False, test=False):
    fm = MNIST('../../data/fashion_mnist')
    sample_examples = []; sample_labels = []; count = 0
    if test:
        images, labels = fm.load_testing()
    else:
        images, labels = fm.load_training()
    for i in range(10):
        idx = 0
        while (count < num_examples/10):
            if labels[idx] == i:
                sample_examples.append(images[idx]); sample_labels.append(labels[idx])
                count += 1
            idx += 1
        count = 0
    if unlabeled:
        return np.array(sample_examples)
    else:
        return np.array(sample_examples), np.array(sample_labels)

def get_mnist(num_examples, unlabeled=False, test=False):
    mn = MNIST('../../data/mnist')
    sample_examples = []; sample_labels = []; count = 0
    if test:
        images, labels = mn.load_testing()
    else:
        images, labels = mn.load_training()
    for i in range(10):
        idx = 0
        while (count < num_examples/10):
            if labels[idx] == i:
                sample_examples.append(images[idx]); sample_labels.append(labels[idx])
                count += 1
            idx += 1
        count = 0
    if unlabeled:
        return np.array(sample_examples)
    else:
        return np.array(sample_examples), np.array(sample_labels)

def get_house_number(num_examples, unlabeled=False, test=False):
    sample_examples = []; sample_labels = []; count = 0
    if test:
        data = scipy.io.loadmat("../../data/house_number/train_32x32.mat")
    else:
        data = scipy.io.loadmat("../../data/house_number/test_32x32.mat")
    images = []
    for i in range(data['X'].shape[3]):
        images.append(imresize(data['X'][:,:,:,i],(28,28)))
    labels = data['y']
    for i in range(len(labels)):
        if labels[i] == 10:
            labels[i] = 0
    for i in range(len(images)):
        images[i] = (rgb2gray(images[i])*256).ravel()
    for i in range(10):
        idx = 0
        while (count < num_examples/10):
            if labels[idx] == i:
                sample_examples.append(images[idx]); sample_labels.append(labels[idx])
                count += 1
            idx += 1
        count = 0
    if unlabeled:
        return np.array(sample_examples)
    else:
        return np.array(sample_examples), np.array(sample_labels)

def get_folds(examples, numfolds, labels=None):
    pass
