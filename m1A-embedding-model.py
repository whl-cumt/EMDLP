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
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.layers import Dense, Dropout,MaxPooling1D,Flatten,Input
from keras.layers import Conv1D,concatenate,LSTM,Bidirectional
from keras.callbacks import EarlyStopping
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
from skopt import Optimizer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
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

feature = np.load('embedding_m1A-101-cz-593-5930_f_train.dat')
label = np.load('embedding_m1A-101-cz-593-5930_l_train.dat')
f_test = np.load('embedding_m1A-101-cz-114-1140_f_test.dat', allow_pickle=True)
l_test= np.load('embedding_m1A-101-cz-114-1140_l_test.dat', allow_pickle=True)

#################5
testAcc1 = 0
testTime1 = 0
seed = 100
kfold = 5
kf = KFold(n_splits = kfold, shuffle=True,random_state = seed)
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
cvspe1 = []
cvsen1 = []
roc_auc_scores_max = 0
thres = 0.9


def ANN(optimizer='adam', neurons=64,kernel_size=5, batch_size=64, epochs=48, activation='relu', patience=6,drop=0.5,
        loss='categorical_crossentropy'):
    inp1 = Input(shape= (99,), dtype='int32')
    inp2 = Embedding(106, 300,input_length=99)(inp1)
    ######### model1
    cnn1 = Conv1D(64, 3, padding='same', strides=1, activation='relu')(inp2)
    cnn3 = MaxPooling1D(pool_size=2)(cnn1)
    drop1 = Dropout(0.2)(cnn3)

    cnn11 = Conv1D(64, 3, padding='same', strides=1, dilation_rate=2, activation='relu')(inp2)
    cnn13 = MaxPooling1D(pool_size=2)(cnn11)
    drop11 = Dropout(0.2)(cnn13)

    cnn21 = Conv1D(64, 3, padding='same', strides=1, dilation_rate=3, activation='relu')(inp2)
    cnn23 = MaxPooling1D(pool_size=2)(cnn21)
    drop21 = Dropout(0.2)(cnn23)
    cnn = concatenate([drop1, drop11, drop21], axis=-1)


    Bi1 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(cnn)  # 0.1

    flat = Flatten()(Bi1)
    x1 = Dense(256, activation='relu')(flat)
    x2 = Dropout(0.5)(x1)
    x3 = Dense(128, activation='relu')(x2)
    x4 = Dropout(0.5)(x3)
    x5 = Dense(64, activation='relu')(x4)
    x6 = Dropout(0.5)(x5)
    final_output = Dense(1, activation='sigmoid')(x6)
    model = Model(inputs=inp1,outputs=final_output)

    model.summary()
    return  model


model=ANN()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callback =EarlyStopping(monitor='val_loss',patience=3,verbose=0, mode='min')
history = model.fit(feature, label, epochs=100, batch_size=32, verbose=2,validation_data=(f_test, l_test),
                       callbacks=[callback])

model.save('m1A-embedding-model.h5')


