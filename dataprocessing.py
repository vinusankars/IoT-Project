import torch
import sys
import os
import numpy as np
import copy
from six.moves import cPickle as pkl
import pandas as pd
import argparse

Keys = {0:'walking', 1:'walking_up', 2:'walking_down', 3:'sitting', 4:'standing', 5:'laying'}
DIR = "/cmlscratch/vinu/HAPT-dataset/RawData"
DATA_DIR = "."
window = 500
overlap = 250

def continuous_to_discrete(signal, labels, pid, window, overlap, test=False):

    X, y, p = [], [], []

    if test == False:
        for j in range(0, len(signal)-window, overlap//4):
            start = j
            end = start + window
            x = signal[start: end]
            count = labels[start: end]
            count = np.stack([(count==i).sum() for i in range(-1,6)])
            l = np.argmax(count)-1
            if l != -1 and (pid[start] == pid[end-1]):
                X.append(x)
                y.append(l)
                p.append(pid[start])

    else:
        for j in range(0, len(signal)-window, overlap):
            x = signal[j: j+window]
            count = labels[j: j+window]
            count = np.stack([(count==i).sum() for i in range(-1,6)])
            l = np.argmax(count)-1
            if l!= -1 and (pid[j] == pid[j+window-1]):
                X.append(x)
                y.append(l)
                p.append(pid[j])

    return torch.tensor(np.stack(X)), torch.tensor(np.stack(y)), torch.tensor(np.stack(p))


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    print(Keys)
    print("Window and overlap is", window, overlap)

    labels = {i: [] for i in range(1, 62)} # keys are expid
    data = open(os.path.join(DIR, 'labels.txt'), 'r').readlines()
    for i in range(len(data)):
        x = data[i].split(' ')
        labels[int(x[0])].append([int(x[2]), int(x[3]), int(x[4][:-1])])

    X = []
    y = []
    p = []

    for i in os.listdir(DIR):
        if i.startswith("acc_exp"):
            f1 = open(os.path.join(DIR, i)).readlines()
            f2 = open(os.path.join(DIR, "gyro"+i[3:])).readlines()
            f = []
            for j in range(len(f1)):
                a = f1[j][:-1].split(' ')
                b = f2[j][:-1].split(' ')
                a.extend(b)
                f.append(list(map(float, a)))
            # del f1, f2 

            l = np.stack([-1]*len(f))
            expid = int(i.split('exp')[1].split('_')[0])
            for j in range(len(labels[expid])):
                if labels[expid][j][0] <= 6:
                    l[labels[expid][j][1]-1: labels[expid][j][2]] = labels[expid][j][0]-1
            X.append(np.stack(f))
            y.append(l)

            user = int(i.split('user')[1].split('.')[0])
            p.append([user]*len(l))

    X_train = np.concatenate(X[:49], axis=0)
    X_test = np.concatenate(X[49:], axis=0)
    y_train = np.concatenate(y[:49], axis=0)
    y_test = np.concatenate(y[49:], axis=0)
    p_train = np.concatenate(p[:49], axis=0)
    p_test = np.concatenate(p[49:], axis=0)

    print("Train data shape is", X_train.shape, y_train.shape)
    print("Test data shape is", X_test.shape, y_test.shape)

    # standardize data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean)/std
    X_test= (X_test - mean)/std
    print("\nMean and std of train data is:\n", mean, std)

    print("\nMax and min values in train data:\n", X_train.max(axis=0), X_train.min(axis=0))

    train_set = continuous_to_discrete(X_train, y_train, p_train, window, overlap, test=False)
    test_set = continuous_to_discrete(X_test, y_test, p_test, window, overlap, test=True)

    print("\nDiscrete data shape:")
    print(train_set[0].shape, train_set[1].shape, train_set[2].shape)
    print(test_set[0].shape, test_set[1].shape, test_set[2].shape)

    print("\nOpening pickle file will give [signal, labels, person_id]")

    with open(os.path.join(DATA_DIR, 'train_set.pkl'), 'wb') as f:
        pkl.dump(train_set, f)

    with open(os.path.join(DATA_DIR, 'test_set.pkl'), 'wb') as f:
        pkl.dump(test_set, f)

    with open(os.path.join(DATA_DIR, 'continuous_test_set.pkl'), 'wb') as f:
        pkl.dump((X_test, y_test), f)