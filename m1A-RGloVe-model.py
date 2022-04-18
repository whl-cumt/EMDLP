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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold,StratifiedKFold
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense, Dropout,MaxPooling1D,Attention,Flatten,Input
from tensorflow.keras.layers import Conv1D,LSTM,Bidirectional,concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from collections import Counter
import re, sys, os
from keras.models import load_model
import random
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
import os
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from scipy import interp
import time
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

splitcharBy=3
overlap_interval=1

window = 2


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


m1A_list_train=list(open(r'm1A_CV.txt','r'))
len_seq = 101
num_in=len(m1A_list_train)
label=[]
feature =[]
#
random.shuffle(m1A_list_train)
for i in range(num_in):
    # seq = str(m1A_list_train[i].seq)
    seq = str(m1A_list_train[i][0:101])
    seq = seq.replace('-', 'X')  # turn rna seq to dna seq if have
    #print(m1A_list_train[i].id)
    TempArray = [seq[j:j + splitcharBy]  for j in range(0, len(seq)-(len(seq) % splitcharBy), overlap_interval)]
    feature.append(TempArray)
    if m1A_list_train[i][-2]=='1':
        label.append(1)
    else:
        label.append(0)


m1A_list_test=list(open(r'm1A_IND.txt','r'))
# len_seq=len(m1A_list_train[0].seq)
len_seq = 101
num_in=len(m1A_list_test)
Y_test=[]  #X_test,Y_test
X_test =[]

random.shuffle(m1A_list_test)
for i in range(num_in):
    # seq = str(m1A_list_train[i].seq)
    seq = str(m1A_list_test[i][0:101])
    seq = seq.replace('-', 'X')  # turn rna seq to dna seq if have
    #print(m1A_list_train[i].id)
    TempArray = [seq[j:j + splitcharBy]  for j in range(0, len(seq)-(len(seq) % splitcharBy), overlap_interval)]
    X_test.append(TempArray)
    if m1A_list_test[i][-2]=='1':
        Y_test.append(1)
    else:
        Y_test.append(0)


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
    return embedding_vectors


wordembedding = load_embedding_vectors("m1A_RGloVe.txt")
feature = np.array(feature)
label = np.array(label)

###########################
def generate_arrays_from_feature(feature, label, batch_size):
    while 1:
        train_data = []
        train_label = []
        cnt = 0
        for i in range(len(feature)):
            temp_list = []
            for j in feature[i]:
                word = j
                if word in wordembedding.keys():
                    temp_list.append(wordembedding[word])
                else:
                    word = "<unk>"
                    temp_list.append(wordembedding[word])
            train_data.append(temp_list)  #
            train_label.append(label[i])
            # print(train_data)
            # print(train_label)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(train_data), np.array(train_label))
                train_data = []
                train_label = []



#################5折
testAcc1 = 0
testTime1 = 0
seed = 100
kfold = 5
kf = KFold(n_splits = kfold, shuffle=True,random_state = seed)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
label_predict = np.zeros(label.shape)
# Loop through the indices the split() method returns
foldi = 0
label_pre = []
cvscores = []
cvroc_auc_score = []
cvmatthews_corrcoef = []
cvaccuracy_score = []
cvprecision_score = []
cvrecall_score = []
cvspe = []
cvsen = []
cvaccuracy_score1 = []
cvprecision_score1 = []
cvmatthews_corrcoef1 = []
# cvroc_auc_score1 = []
cvspe1 = []
cvsen1 = []
roc_auc_scores_max = 0
thres = 0.9

batch_size=32
epochs=100

def ANN(optimizer='adam', neurons=64,kernel_size=5, batch_size=64, epochs=60, activation='relu', patience=50,drop=0.2,
        loss='categorical_crossentropy'):
    inp1 = Input(shape= (99,300), dtype='float32')
    ######### model1
    cnn1 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(inp1)
    cnn3 = MaxPooling1D(pool_size=2)(cnn1)
    drop1 = Dropout(0.2)(cnn3)

    cnn11 = Conv1D(64, 3, padding='same', strides=1, dilation_rate=2,activation='relu')(inp1)
    cnn13 = MaxPooling1D(pool_size=2)(cnn11)
    drop11 = Dropout(0.2)(cnn13)

    cnn21 = Conv1D(64, 3, padding='same', strides=1, dilation_rate=3,activation='relu')(inp1)
    cnn23 = MaxPooling1D(pool_size=2)(cnn21)
    drop21 = Dropout(0.2)(cnn23)
    cnn = concatenate([drop1,  drop11,drop21], axis=-1)


    Bi1 = Bidirectional(LSTM(64,dropout=0.2, recurrent_dropout=0.2,return_sequences = True))(cnn)#

    flat = Flatten()(Bi1)
    x1 = Dense(256, activation='relu')(flat)
    x2 = Dropout(0.5)(x1)
    x3 = Dense(128, activation='relu')(x2)
    x4 = Dropout(0.5)(x3)
    x5 = Dense(64, activation='relu')(x4)
    x6 = Dropout(0.5)(x5)
    final_output = Dense(1, activation='sigmoid')(x6)
    model = Model(inputs =inp1,outputs = final_output)      #


    model.summary()

    return  model


model = ANN()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=3e-4),
              metrics=['accuracy'])
#
earlystopping=EarlyStopping(monitor='val_loss',patience=3,verbose=0, mode='min')
callbacks = [earlystopping]
model.fit_generator(generate_arrays_from_feature(feature,label, batch_size),  # fit_generator 
                    epochs=epochs,
                    steps_per_epoch=len(feature) // batch_size,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=generate_arrays_from_feature(X_test,Y_test, batch_size),
                    validation_steps=len(X_test) // batch_size)
####save
model.save('m1A-RGloVe-model.h5')



