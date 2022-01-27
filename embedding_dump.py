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
import collections
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


splitcharBy=3
overlap_interval=1

m1a_list_train=list(open(r'm1A_CV.txt','r'))
len_seq = 101
num_in=len(m1a_list_train)
label=[]
feature =[]
random.shuffle(m1a_list_train)
for i in range(num_in):
    seq = str(m1a_list_train[i][0:101])
    #print(m1a_list_train[i].id)
    TempArray = [seq[j:j + splitcharBy]  for j in range(0, len(seq)-(len(seq) % splitcharBy), overlap_interval)]
    feature.append(TempArray)

    if m1a_list_train[i][-2]=='1':
        label.append(1)
    else:
        label.append(0)

m1a_list_test=list(open(r'm1A_IND.txt','r'))
len_seq = 101
num_inT=len(m1a_list_test)
l_test=[]
f_test =[]
random.shuffle(m1a_list_test)
for i in range(num_inT):
    seq = str(m1a_list_test[i][0:101])
    TempArray = [seq[j:j + splitcharBy]  for j in range(0, len(seq)-(len(seq) % splitcharBy), overlap_interval)]
    f_test.append(TempArray)

    if m1a_list_test[i][-2]=='1':
        l_test.append(1)
    else:
        l_test.append(0)


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = reader.readline()
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab

def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output

def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)

vocab=load_vocab('genome.txt')


a=[]
for i in range(len(feature)):
    #print(feature[i])
    a.append(convert_tokens_to_ids(vocab, feature[i]))
feature=np.array(a)
label=np.array(label) 


feature = feature.reshape((num_in,len(seq)-(len(seq) % splitcharBy)))    #
feature.dump("embedding_m1A-101-cz-593-5930_f_train.dat")
label.dump("embedding_m1A-101-cz-593-5930_l_train.dat")


b=[]
for i in range(len(f_test)):
    #print(feature[i])
    b.append(convert_tokens_to_ids(vocab, f_test[i]))
f_test=np.array(b)
l_test=np.array(l_test)


f_test = f_test.reshape((num_inT,len(seq)-(len(seq) % splitcharBy))) #
f_test.dump("embedding_m1A-101-cz-114-1140_f_test.dat")
l_test.dump("embedding_m1A-101-cz-114-1140_l_test.dat")

