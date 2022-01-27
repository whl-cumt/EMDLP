import numpy as np
import gensim
import keras
import tensorflow as tf
from Bio import SeqIO
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import  accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.layers import Dense, Dropout,MaxPooling1D,Flatten
from keras.layers import Conv1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from collections import Counter
import re, sys, os
from keras.models import load_model
import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from numpy import array
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

m1a_list_train=list(open(r'm1A_CV.txt','r'))
len_seq = 101
num_in = len(m1a_list_train)
label = []
feature = []
arr = []
random.shuffle(m1a_list_train)
encoder = LabelBinarizer()
for i in range(num_in):
    seq = str(m1a_list_train[i][0:101])
    for c in seq:
        arr.append(c)

    if m1a_list_train[i][-2]=='1':
        label.append(1)
    else:
        label.append(0)
values = array(arr)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
feature = onehot_encoder.fit_transform(integer_encoded)
feature = np.array(feature)
label = np.array(label)
feature = feature.reshape((num_in,101,5))
feature.dump("onehot_m1A-101-cz-593-5930_f_train.dat")
label.dump("onehot_m1A-101-cz-593-5930_l_train.dat")


m1a_list_test=list(open(r'm1A_IND.txt','r'))
len_seq = 101
num_in = len(m1a_list_test)
l_test = []
f_test = []
arr = []
random.shuffle(m1a_list_test)
encoder = LabelBinarizer()
for i in range(num_in):
    seq = str(m1a_list_test[i][0:101])
    for c in seq:
        arr.append(c)

    if m1a_list_test[i][-2]=='1':
        l_test.append(1)
    else:
        l_test.append(0)

values_test = array(arr)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values_test)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
f_test = onehot_encoder.fit_transform(integer_encoded)
f_test = np.array(f_test)
l_test = np.array(l_test)
f_test = f_test.reshape((num_in,101,5))
f_test.dump("onehot_m1A-101-cz-114-1140_f_test.dat")
l_test.dump("onehot_m1A-101-cz-114-1140_l_test.dat")