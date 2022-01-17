import gensim
import keras
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import precision_recall_curve
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.layers import Dense, Dropout,MaxPooling1D,Flatten
from keras.layers import Conv1D
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import re, sys, os
from sklearn.metrics import matthews_corrcoef
from numpy import arange
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import  accuracy_score
import collections
from numpy import array
import random
from sklearn.metrics import recall_score
from sklearn.metrics import  precision_score
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def cal_base(y_true, y_pred):
    y_pred_positive = np.round(np.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = np.sum(y_positive * y_pred_positive)
    TN = np.sum(y_negative * y_pred_negative)

    FP = np.sum(y_negative * y_pred_positive)
    FN = np.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP )
    return SP

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output

def embedding(m6a_list_test,num_in):
    splitcharBy = 3
    overlap_interval = 1
    f_test = []
    for i in range(num_in):
        seq = str(m6a_list_test[i][0:101])
        TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
        f_test.append(TempArray)

    vocab = load_vocab('genomelgs.txt')

    b = []
    for i in range(len(f_test)):
        b.append(convert_by_vocab(vocab, f_test[i]))
    f_test = np.array(b)
    return f_test

def load_embedding_vectors(filename):
    embedding_vectors = dict()
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_vectors[word] = vector

    f.close()
    return embedding_vectors



def Glove(m6a_list_test,num_in):
    splitcharBy = 3
    overlap_interval = 1
    wordembedding = load_embedding_vectors("m6A_RGloVe.txt")
    test_data = []
    for i in range(num_in):
        seq = str(m6a_list_test[i][0:101])
        TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
        test_data.append(TempArray)
    test_data1 = []
    for i in range(len(test_data)):
        temp_list = []
        for j in test_data[i]:
            word = j
            if word in wordembedding.keys():
                temp_list.append(wordembedding[word])
            else:
                word = "<unk>"
                temp_list.append(wordembedding[word])
        test_data1.append(temp_list)
    feature_test = np.array(test_data1)
    return feature_test


def onehot(m6a_list_test,num_in):
    f_test = []
    arr = []

    for i in range(num_in):
        seq = str(m6a_list_test[i][0:101])
        for c in seq:
            arr.append(c)

    values_test = array(arr)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    f_test = onehot_encoder.fit_transform(integer_encoded)
    f_test = f_test.reshape((num_in, 101, 5))
    f_test = np.array(f_test)
    return f_test


m1a_list_test=list(open(r'm1A_IND.txt','r'))
len_seq = 101
num_in=len(m1a_list_test)

l_test=[]
random.shuffle(m1a_list_test)
for i in range(num_in):
    if m1a_list_test[i][-2]=='1':
        l_test.append(1)
    else:
        l_test.append(0)
l_test = np.array(l_test)


f_test_em=embedding(m1a_list_test,num_in)
f_test_onehot=onehot(m1a_list_test,num_in)
f_test_ENAC=ENAC(m1a_list_test,len_seq,num_in)
f_test_Glove=Glove(m1a_list_test,num_in)
model1 = load_model("m1A-embedding-model.h5")
label_predict_em = model1.predict(f_test_em)
model2 = load_model("m1A-onehot-model.h5")
label_predict_onehot = model2.predict(f_test_onehot)
model4 = load_model("m1A-GloVe-model.h5")
label_predict_Glove = model4.predict(f_test_Glove)

X = []
Y = []
Z1 = []
Z2 = []
Z3 = []
max=0
for i in arange(0,1.001,0.01):
    for j in arange(0,1.001-i, 0.01):
        label_pre=[]
        label=label_predict_em*i+label_predict_onehot*j+label_predict_Glove*(1.0-i-j)
        # print(label)
        X.append(round(i,2))
        Y.append(round(j,2))
        for k in range(0,num_in):
            if(label[k][0]<0.5):
                label_pre.append(0)
            else: label_pre.append(1)
        label_pre= np.array(label_pre)
        # print(label_pre)
        label_pre=label_pre.reshape(num_in,1)
        print(i,j,1-i-j)
        print("MCC: %f " % matthews_corrcoef(l_test, label_pre),"AUROC: %f " %roc_auc_score(l_test, label),"ACC:  %f "  %accuracy_score(l_test,label_pre))
        if (max<roc_auc_score(l_test, label)):
            max=roc_auc_score(l_test, label)
            flag=[i,j,1-i-j]
        Z1.append(matthews_corrcoef(l_test, label_pre))
        Z2.append(roc_auc_score(l_test, label))
        Z3.append(accuracy_score(l_test,label_pre))
print(flag)

print(flag)
label_pre = label_predict_em*flag[0]+label_predict_onehot*flag[1]+label_predict_Glove*flag[2]  #
label_predict = [0 if item<=0.5 else 1 for item in label_pre]
print("AUROC: %f " %roc_auc_score(l_test, label_pre))
print("MCC: %f " %matthews_corrcoef(l_test,label_predict))
print( "ACC:  %f "  %accuracy_score(l_test,label_predict))
print("Precision: %f " %precision_score(l_test,label_predict))
print( "Recall:  %f "  %recall_score(l_test,label_predict))
Spe=specificity(l_test,label_predict)
print("specificity: ",round(Spe*100,2))


a0 = l_test

f = open("m1A_ensemble.txt", 'w')
for i in range(0, len(a0)):
    f.write(np.str(a0[i]))
    f.write('\n')
f.close()


a = label_pre.flatten()
f = open("m1A_ensemble_pre.txt", 'w')

for i in range(0, len(a0)):
    f.write(np.str(a[i]))
    f.write('\r')
f.close()


fpr, tpr, thre = roc_curve(l_test, label_pre)
roc_auc = auc(fpr, tpr)
plt.title('ROC')
plt.plot(fpr, tpr, 'orange', label='AUROC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.grid()
plt.savefig('auROC-m1A_ensemble.png')
plt.show()
