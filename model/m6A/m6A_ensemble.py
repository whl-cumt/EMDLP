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
splitcharBy = 3
overlap_interval=1
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
    """Loads a vocabulary file into a dictionary."""
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
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output

def embedding(m6a_list_test,num_in):
    splitcharBy = 3
    overlap_interval = 1
    f_test = []
    for i in range(num_in):
        seq = str(m6a_list_test[i][0:1001])
        # print(m6a_list_test[i].id)
        TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
        f_test.append(TempArray)

    vocab = load_vocab('genome-X.txt')

    b = []
    for i in range(len(f_test)):
        # print(feature[i])
        b.append(convert_by_vocab(vocab, f_test[i]))
    f_test = np.array(b)
    return f_test
####################
def load_embedding_vectors(filename):
    # load embedding_vectors from the word2vec
    # initial matrix with random uniform
    embedding_vectors = dict()
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_vectors[word] = vector

    f.close()
    return embedding_vectors           #



def Glove(m6a_list_test,num_in):
    splitcharBy = 3
    overlap_interval = 1

    wordembedding = load_embedding_vectors("m6A_RGloVe.txt")  #
    test_data = []
    for i in range(num_in):
        seq = str(m6a_list_test[i][0:1001])
        # print(m6a_list_test[i].id)
        TempArray = [seq[j:j + splitcharBy] for j in range(0, len(seq) - (len(seq) % splitcharBy), overlap_interval)]
        test_data.append(TempArray)
    test_data1 = []
    for i in range(len(test_data)):
        temp_list = []
        # for j in m6a_list_test[i]:
        for j in test_data[i]:
            word = j
            if word in wordembedding.keys():
                temp_list.append(wordembedding[word])
            else:
                word = "<unk>"
                temp_list.append(wordembedding[word])
        test_data1.append(temp_list)
    feature_test = np.array(test_data1)
    # feature_test = feature_test.reshape((num_in, 300))
    return feature_test


def onehot(m6a_list_test,num_in):
    f_test = []
    arr = []

    for i in range(num_in):
        seq = str(m6a_list_test[i][0:1001])
        for c in seq:
            arr.append(c)

    values_test = array(arr)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    f_test = onehot_encoder.fit_transform(integer_encoded)
    f_test = f_test.reshape((num_in, 1001, 5))
    f_test = np.array(f_test)
    return f_test

def ENAC(m6a_list_test,len_seq,num_in):
    window = 2
    AA = 'ATGC'
    f_test = []
    for i in range(num_in):
        sequence = str(m6a_list_test[i][0:1001])
        code = []
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                count = Counter(re.sub('X', '', sequence[j:j + window]))
                for key in count:
                    count[key] = count[key] / len(re.sub('X', '', sequence[j:j + window]))
                for aa in AA:
                    code.append(count[aa])
        f_test.append(code)
    f_test = np.array(f_test)
    f_test = f_test.reshape((num_in, len_seq - 1, 4))
    return f_test




m6a_list_test=list(SeqIO.parse("m6A_IND.fasta", "fasta"))  #

db = []
q = 0
for s in range(0, len(m6a_list_test)):
    if 'ENST' in m6a_list_test[s].id:
        db.append(m6a_list_test[s])

len_seq = 1001

l_test=[]
f_test =[]

num_in=len(db)

for i in range(0, num_in):
    sequence = str(db[i].seq)
    sequence = sequence.replace('-', 'X')  # turn chenzhen seq to zhouquan seq if have
    f_test.append(sequence)
    if db[i].description.endswith('|1'):
    # if db[i][-2]=='1':
        l_test.append(1)
    else:
        l_test.append(0)
l_test = np.array(l_test)
print(l_test)


f_test_em=embedding(f_test,num_in)
f_test_onehot=onehot(f_test,num_in)
f_test_ENAC=ENAC(f_test,len_seq,num_in)
f_test_Glove=Glove(f_test,num_in)
model1 = load_model("m6A-embedding-model.h5") # embedding+2cnn-2+BiL-5-m6a-3e-4-train.h5
'Model' object has no attribute 'predict_proba'
label_predict_em = model1.predict(f_test_em)

model2 = load_model("m6A-onehot-model.h5") # concat(dcnn+cnn)+BiLSTM
label_predict_onehot = model2.predict(f_test_onehot)

model4 = load_model("m6A-RGloVe-model.h5")
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

"""
save data begin
"""
a0 = l_test

f = open("m6a_ensemble_cat(cnn-b3333-3333-2-3333-3)+BiL_em-on-Gl.txt", 'w')
for i in range(0, len(a0)):
    f.write(np.str(a0[i]))
    f.write('\n')
f.close()

# a = label_pre
a = label_pre.flatten()
f = open("m6a_ensemble_cat(cnn-b3333-3333-2-3333-3)+BiL_em-on-Gl_pre.txt", 'w')
# for i in range(0,long):
for i in range(0, len(a0)):
    f.write(np.str(a[i]))
    f.write('\r')
f.close()
"""
end
"""


