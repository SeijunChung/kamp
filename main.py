# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.pyplot import cm
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import itertools

from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding,LSTM, Flatten, MaxPooling1D, ReLU, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv1D, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"]="0"


save_path = "./results"
file_path = './data'  # 다운로드 받은 데이터 파일이 위치한 경로(file path)
train_fn="/FordA_TRAIN.arff"  # Train 데이터 파일명
test_fn="/FordA_TEST.arff"  # Test 데이터 파일명

def read_ariff(path):
    """
    .ariff 확장자를 Load하기 위한 함수
    """
    raw_data, meta = loadarff(path)
    cols = [x for x in meta]
    data2d = np.zeros([raw_data.shape[0],len(cols)])
    for i,col in zip(range(len(cols)),cols):
        data2d[:,i]=raw_data[col]
    return data2d

train = read_ariff(file_path + train_fn)
test = read_ariff(file_path + test_fn)

print("train_set.shape:", train.shape)
print("test_set.shape:", test.shape)


x_train_temp = train[:,:-1]
y_train_temp = train[:, -1]  # 마지막 column이 Label 값이 있는 column
x_test = test[:, :-1]
y_test = test[:, -1]  # 마지막 column이 Label 값이 있는 column

normal_x = x_train_temp[y_train_temp==1]  # Train_x 데이터 중 정상 데이터
abnormal_x = x_train_temp[y_train_temp==-1]  # Train_x 데이터 중 비정상 데이터
normal_y = y_train_temp[y_train_temp==1]  # Train_y 데이터 중 정상 데이터
abnormal_y = y_train_temp[y_train_temp==-1]  # Train_y 데이터 중 비정상 데이터

ind_x_normal = int(normal_x.shape[0]*0.8)  # train_x 데이터를 8:2로 나누기 위한 기준 인덱스 설정
ind_y_normal = int(normal_y.shape[0]*0.8)  # train_y 데이터를 8:2로 나누기 위한 기준 인덱스 설정
ind_x_abnormal = int(abnormal_x.shape[0]*0.8)  # train_x 데이터를 8:2로 나누기 위한 기준 인덱스 설정
ind_y_abnormal = int(abnormal_y.shape[0]*0.8)  # train_y 데이터를 8:2로 나누기 위한 기준 인덱스 설정

x_train = np.concatenate((normal_x[:ind_x_normal], abnormal_x[:ind_x_abnormal]), axis=0)
x_valid = np.concatenate((normal_x[ind_x_normal:], abnormal_x[ind_x_abnormal:]), axis=0)
y_train = np.concatenate((normal_y[:ind_y_normal], abnormal_y[:ind_y_abnormal]), axis=0)
y_valid = np.concatenate((normal_y[ind_y_normal:], abnormal_y[ind_y_abnormal:]), axis=0)

print("x_train.shape:", x_train.shape)
print("x_valid.shape:", x_valid.shape)
print("y_train.shape:", y_train.shape)
print("y_valid.shape:", y_valid.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

""" 데이터 시각화

시각화 I: Data Imbalance 여부 확인
"""

# Class의 종류 확인: 정상 1, 비정상 -1
classes = np.unique(np.concatenate((y_train, y_test), axis=0))  # classes = array([-1,  1])

x = np.arange(len(classes))  # Plot의 X축의 개수 구하기
labels = ["Abnormal", "Normal"]   # Plot의 X축의 이름 구하기

values_train = [(y_train == i).sum() for i in classes]  # Train 데이터의 정상/비정상 각 총 개수
values_valid = [(y_valid == i).sum() for i in classes]  # Test 데이터의 정상/비정상 각 총 개수
values_test = [(y_test == i).sum() for i in classes]  # Test 데이터의 정상/비정상 각 총 개수

plt.figure(figsize=(8,4))  # Plot 틀(Figure)의 Size 설정 (5X3)

plt.subplot(1,3,1)   # Plot 틀(Figure) 내 3개의 subplot 중 첫 번째(왼쪽) 지정
plt.title("Training Data")  # subplot 제목
plt.bar(x, values_train, width=0.6, color=["red", "blue"])  # Train 데이터의 정상/비정상 개수 BarPlot
plt.ylim([0, 1500])
plt.xticks(x, labels)  # X축에 변수 기입

plt.subplot(1,3,2)  # Plot 틀(Figure) 내 3개의 subplot 중 두 번째(가운데) 지정
plt.title("Validation Data")
plt.bar(x, values_valid, width=0.6, color=["red", "blue"])  # Test 데이터의 정상/비정상 개수 BarPlot
plt.ylim([0, 1500])
plt.xticks(x, labels)  

plt.subplot(1,3,3)  # Plot 틀(Figure) 내 3개의 subplot 중 세 번째(오른쪽) 지정
plt.title("Test Data")
plt.bar(x, values_test, width=0.6, color=["red", "blue"])  # Test 데이터의 정상/비정상 개수 BarPlot
plt.ylim([0, 1500])
plt.xticks(x, labels)

plt.tight_layout()  # 그림 저장
plt.savefig(save_path + '/plots/data_imbalance.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()  # 그림 출력

"""시각화 II: 1개의 timeseries example을 Plot"""

import random

labels = np.unique(np.concatenate((y_train, y_test), axis=0))  # labels (-1 or 1)

plt.figure(figsize = (10, 4))
for c in labels:
    c_x_train = x_train[y_train == c]
    if c == -1: c = c + 1  # 편의 상 Abnormal Class(-1)를 0으로 조정
    time_t = random.randint(0, c_x_train.shape[0]) # 0~1404 사이의 랜덤한 정수가 특정 time t가 됨
    plt.plot(range(0, 500), c_x_train[time_t], label="Normal" if c == 0 else "Abnormal")
plt.legend(loc="upper right", fontsize=13)
plt.xlabel("Time", fontsize=13)
plt.ylabel("Sensor Value", fontsize=13)
plt.savefig(save_path + '/plots/ford_a.png', dpi=100, bbox_inches='tight')
plt.show()
plt.close()

def get_scatter_plot(c):
    time_t = random.randint(0, c_x_train.shape[0])  # 0~1404 사이의 랜덤한 정수가 특정 time t가 됨
    plt.scatter(range(0, c_x_train.shape[1]), c_x_train[time_t], 
                marker='o', s=5, c="r" if c == -1  else "b")
    plt.title("at time: t_{}".format(time_t), fontsize=20)
    plt.xlabel("Sensor", fontsize=14)
    plt.ylabel("Sensor Value", fontsize=14)
    plt.savefig(save_path + '/plots/state.png'.format(state="abnormal" if c == -1 else "normal"),
                dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

labels = np.unique(np.concatenate((y_train, y_test), axis=0))

for c in labels:
    c_x_train = x_train[y_train == c]
    if c == -1:
        print("비정상 Label 데이터 수: ", len(c_x_train))
        get_scatter_plot(c)
    else:
        print("정상 Label 데이터 수: ", len(c_x_train))
        get_scatter_plot(c)

"""시각화 II: 1개의 Sensor 값의 Flow를 Plot"""

sensor_number = random.randint(0, 500)  # 0~500 사이의 랜덤한 정수가 Sensor 번호가 됨

plt.figure(figsize = (13, 4))
plt.title("sensor_number: {}".format(sensor_number), fontsize=20)
plt.plot(x_train[:, sensor_number])
plt.xlabel("Time", fontsize=15)
plt.ylabel("Sensor Value", fontsize=15)
plt.savefig(save_path + '/plots/ford_a_sensor.png', dpi=100, bbox_inches='tight')
plt.show()
plt.close()

# from matplotlib import cm as cm
# from matplotlib.collections import EllipseCollection
# import seaborn as sns

# df = pd.DataFrame(data = x_train, columns= ["sensor_{}".format(label+1) for label in range(x_train.shape[1])])

# data = df.corr()

# sns.set(rc = {'figure.figsize':(12,8)})
# sns.heatmap(data, 
#                annot = False,      # 실제 값 화면에 나타내기
#                cmap = 'bwr',  # Red, Yellow, Blue 색상으로 표시
#                vmin = -1, vmax = 1, #컬러차트 -1 ~ 1 범위로 표시
#             xticklabels=False, yticklabels=False
#               )
# plt.tight_layout()
# plt.savefig(save_path + 'corr.png', dpi=100, bbox_inches='tight')  # 그림 저장

from matplotlib import cm as cm
from matplotlib.collections import EllipseCollection

df = pd.DataFrame(data = x_train, columns= ["sensor_{}".format(label+1) for label in range(x_train.shape[1])])

data = df.corr()


def plot_corr_ellipses(data, ax =None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cmap = cm.get_cmap('jet', 31)
m = plot_corr_ellipses(data, ax=ax, cmap=cmap)
cb = fig.colorbar(m)
cb.set_label('Correlation coefficient')
# ax.margins(0.1)

plt.title('Correlation between ti')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.tight_layout()
# plt.savefig(save_path + 'corr.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()

""" 데이터 정규화"""

# Standard Scaler를 적용하고 싶을 경우 아래 Code를 실행

stder = StandardScaler()
stder.fit(x_train)
x_train = stder.transform(x_train)
x_valid = stder.transform(x_valid)

# Robust Scaler를 적용하고 싶을 경우 아래 Code를 실행

# rscaler = RobustScaler() 
# rscaler.fit(x_train)
# x_train = rscaler.transform(x_train)
# x_valid = rscaler.transform(x_valid)

trainx = np.expand_dims(x_train, -1)
validx = np.expand_dims(x_valid, -1)
testx = np.expand_dims(x_test, -1)
print("x_train의 형태:", trainx.shape)
print("x_valid의 형태:", validx.shape)
print("x_test의 형태:", testx.shape)

y_train[y_train == -1] = 0
y_valid[y_valid == -1] = 0
y_test[y_test == -1] = 0

num_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

"""## 모델 구축

## 1. CNN
Fully Convolutional Neural Network 을 설계하며, 본 모델은 다음 
[논문](https://arxiv.org/abs/1611.06455)을 참조하였다.
본 implementation은 Tensorflow 2.0을 기준으로 작성하였다.
"""

conv1_channel, conv2_channel, conv3_channel = 64, 64, 64

conv1_size, conv2_size, conv3_size = 3, 3, 3
conv1_pad, conv2_pad, conv3_pad = int((conv1_size -1) /2),int((conv2_size -1) /2),int((conv3_size -1) /2)
conv1_stride, conv2_stride, conv3_stride = 1, 1, 1

pool1_size, pool2_size, pool3_size = 2, 2, 2
pool1_pad, pool2_pad, pool3_pad = 0, 0, 0
pool1_stride, pool2_stride, pool3_stride = 2, 2, 2

def make_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=conv1_channel, kernel_size=conv1_size, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU(name='relu1'))
    model.add(MaxPooling1D(pool_size=pool1_size, strides=pool1_stride, padding='same', name='pool1'))
    model.add(Conv1D(filters=conv2_channel, kernel_size=conv2_size, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU(name='relu2'))
    model.add(MaxPooling1D(pool_size=pool2_size, strides=pool2_stride, padding='same', name='pool2'))
    model.add(Conv1D(filters=conv3_channel, kernel_size=conv3_size, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU(name='relu3'))
    model.add(MaxPooling1D(pool_size=pool3_size, strides=pool3_stride, padding='same', name='pool3'))
    model.add(GlobalAveragePooling1D(name='pool4'))
    model.add(Dense(2, activation="softmax"))
    return model

model = make_cnn_model()

"""## 학습(Training the Model)"""

epochs = 300
batch_size = 64

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        save_path + "/models/cnn_best_model.h5", save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

with tf.device("/device:GPU:7"):
    history = model.fit(
        trainx,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(validx, y_valid),
        verbose=1,
    )

model.summary()

"""## 테스트 데이터셋으로 학습된 Model을 평가"""

model = tf.keras.models.load_model(save_path + "/models/cnn_best_model.h5")
scores = model.evaluate(x_test, y_test)

print("\n""Test accuracy", scores[1])
print("\n""Test loss", scores[0])

"""## 분석 결과값 도출 (시각화)

1. Confusion Matrix
"""

def draw_confusion_matrix(model, xt, yt, model_name):
    Y_pred = model.predict(xt)
    if model_name in ["cnn", "rnn"]:
        y_pred = np.argmax(Y_pred, axis=1)
    else: y_pred = Y_pred
    plt.figure(figsize=(3,3))
    cm = confusion_matrix(yt, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['False', 'True'], rotation=45)
    plt.yticks(tick_marks, ['False', 'True'])
    thresh = cm.max()/1.2
    normalize = False
    fmt = '.2f' if normalize else 'd'
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j], fmt), 
                 horizontalalignment="center", 
                 color="white" if cm[i,j] > thresh else "black", 
                 fontsize=12)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path + '/plots/{}_cm.png'.format(model_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()
    print(classification_report(yt, y_pred))

draw_confusion_matrix(model, x_test, y_test, "cnn")


def draw_roc(model,xt, yt, model_name):
    Y_pred = model.predict(xt)
    if model_name in ["cnn", "rnn"]:
        y_pred = np.argmax(Y_pred, axis=1)
    else: y_pred = Y_pred
    fpr, tpr, thr = roc_curve(yt, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic {};'.format(model_name))
    plt.legend(loc="lower right")
    plt.ion()
    plt.tight_layout()
    plt.savefig(save_path + '/plots/{}_cm.png'.format(model_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()
  
draw_roc(model, x_test, y_test, "cnn")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history["val_loss"])
plt.title("Training & Validation Loss")
plt.ylabel("loss", fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "validation"], loc="best", fontsize=13)
plt.tight_layout()
plt.savefig(save_path + '/plots/cnn_loss.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()
plt.close()

plt.figure()
plt.plot(history.history["sparse_categorical_accuracy"])
plt.plot(history.history["val_" + "sparse_categorical_accuracy"])
plt.title("Model " + "Prediction Accuracy")
plt.ylabel("sparse_categorical_accuracy", fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "validation"], loc="lower right", fontsize=13)
plt.tight_layout()
plt.savefig(save_path + '/plots/cnn_prediction.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()
plt.close()


def nextlen(l, size, pad, stride):
    return int((l+2*pad-size+stride)/stride ) 

input_len = trainx.shape[1]
l1_conv_len = nextlen(input_len, conv1_size, conv1_pad, conv1_stride)
l1_pool_len = nextlen(l1_conv_len, pool1_size, pool1_pad, pool1_stride) +1
l2_conv_len = nextlen(l1_pool_len, conv2_size, conv2_pad, conv2_stride)
l2_pool_len = nextlen(l2_conv_len, pool2_size, pool2_pad, pool2_stride) +1
l3_conv_len = nextlen(l2_pool_len, conv3_size, conv3_pad, conv3_stride)
l3_pool_len = nextlen(l3_conv_len, pool3_size, pool3_pad, pool3_stride) +1
print(l1_conv_len,l1_pool_len,l2_conv_len,l2_pool_len,l3_conv_len,l3_pool_len)

Receptive_Field_index3 = []
for node in range(input_len):
    l3_conv_start = node*pool3_stride - pool3_pad
    l3_conv_end = node*pool3_stride - pool3_pad + pool3_size - 1
    l2_result_start = l3_conv_start*conv3_stride - conv3_pad
    l2_result_end = l3_conv_end*conv3_stride - conv3_pad + conv3_size -1    
    l2_conv_start = l2_result_start*pool3_stride - pool2_pad
    l2_conv_end = l2_result_end*pool2_stride - pool2_pad + pool2_size - 1
    l1_result_start = l2_conv_start*conv2_stride - conv2_pad
    l1_result_end = l2_conv_end*conv2_stride - conv2_pad + conv2_size -1
    l1_conv_start = l1_result_start*pool1_stride - pool1_pad
    l1_conv_end = l1_result_end*pool1_stride - pool1_pad + pool1_size - 1
    input_start = l1_conv_start*conv1_stride - conv1_pad
    input_end = l1_conv_end*conv1_stride - conv1_pad + conv1_size -1
    Receptive_Field_index3.append(np.arange(input_start,input_end+1))

len(Receptive_Field_index3)

layer_name = 'pool3'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(trainx)

percent = 1  ### top 5%

threshold = np.percentile(intermediate_output[:, :, :],100-percent, [0,1])
threshold_bool = (intermediate_output[:, :, :] > threshold)

pattern_length = np.max([len(x) for x in Receptive_Field_index3])  # 277

reindex_threshold_bool=[]
for data_idx in range(2880):  # 2880
    for output_c in range(conv3_channel):  # 128
        if len([x for x in threshold_bool[data_idx,:,output_c] if x]):
            index = []
            for idx in [i for i,x in enumerate(threshold_bool[data_idx,:,output_c]) if x]:    
                reindex_threshold_bool.append([data_idx,output_c,idx])

pattern_idx_df = pd.DataFrame(reindex_threshold_bool,columns=["data_idx","output_channel","pattern_xs"])
groups = pattern_idx_df.groupby(["data_idx","pattern_xs"]) #["output_channel"].apply(list)
pattern_repetitive_idx_df = groups["output_channel"].apply(list).reset_index(name='output_channel')

TAP=[]
pattern_id=0

data_idx = pattern_repetitive_idx_df.loc[:,"data_idx"].values.tolist()
output_channel = pattern_repetitive_idx_df.loc[:,"output_channel"].values.tolist()
pattern_xs = pattern_repetitive_idx_df.loc[:,"pattern_xs"].values.tolist()

for d_idx, output_c, p_xs in zip(data_idx, output_channel, pattern_xs):
    for input_c in range(trainx.shape[2]): # 128
        if (Receptive_Field_index3[p_xs][0] >= 0) and (Receptive_Field_index3[p_xs][-1] < trainx.shape[1]) :
            pattern_dict={}
            pattern_dict["pattern_id"]= pattern_id
            pattern_dict["data_idx"] = d_idx
            pattern_dict["output_channel"] = output_c
            pattern_dict["input_channel"] = input_c
            pattern_dict["pattern_xs"]= Receptive_Field_index3[p_xs]
            pattern_dict["pattern_ys"] = trainx[d_idx, Receptive_Field_index3[p_xs], input_c]
            pattern_dict["features"] = intermediate_output[d_idx, p_xs, :]
            pattern_dict["activations"] = threshold_bool[d_idx, p_xs, :]

            TAP.append(pattern_dict)
            pattern_id += 1


subsequences = np.array([x['pattern_ys'] for x in TAP])
display_n= 32

column = 4
row = int((display_n-1)/column)+1

fig = plt.figure(figsize=(10,row*1.5))
gs = gridspec.GridSpec(row,column)
gs.update(wspace=0, hspace=0)

samples = np.random.choice(len(subsequences), display_n, replace=False)

for i, n in enumerate(samples):
    ax = plt.subplot(gs[i])
    ax.plot(subsequences[n], color='black', alpha=1, linewidth=1.5)
    ax.set_ylim(-5,5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax.spines.values(), linewidth=2)
# plt.suptitle("Patterns Before Clustering", y=1.05, fontsize=18)
plt.tight_layout()
plt.savefig(save_path + '/plots/subsequence.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()

def select_greedy_protos(K, m):

    ''' selected: an array of selected prototypes '''
    ''' obj_list: a list of the objective values '''

    n = np.shape(K)[0]
    selected = np.array([], dtype=int)
    obj_list = []
    nsk = 0

    colsum = 2/n*np.sum(K, axis=0)

    for i in range(m):
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)
        vec1 = colsum[candidates]
        lenS = len(selected)

        if lenS > 0:
            temp = K[selected, :][:, candidates]
            vec2 = np.sum(temp, axis=0) *2 + np.diagonal(K)[candidates]
            vec2 = vec2/(lenS + 1)
            vec3 = vec1 - vec2
        else:
            vec3 = vec1 - (np.abs(np.diagonal(K)[candidates]))

        ''' vec3: {J(selected U {new})-J(selected)}*(lenS + 1) '''
        ''' increase of the objective value'''
        max_idx = np.argmax(vec3)

        if lenS > 0:
            ''' j: J(selected U {new})'''
            sk = np.sum(K[selected, :][:, selected])
            j = vec3[max_idx]/(lenS+1) - nsk/(lenS*(lenS+1)) + (1/(lenS**2)-1/((lenS+1)**2))*sk
            obj_list.append(j)
        else:
            obj_list.append(vec3[max_idx])

        argmax = candidates[max_idx]
        selected = np.append(selected, argmax)

        ''' nsk: (2/n)*\sum{k([n],S)} '''
        nsk += vec1[max_idx]

    return selected, obj_list

data_indices = np.array([x['data_idx'] for x in TAP])
subsequences = np.array([x['pattern_ys'] for x in TAP])
features = np.array([x['features'] for x in TAP])

# Commented out IPython magic to ensure Python compatibility.
# %%time
powers = 20
pat = np.array(features)
pat = pat/(np.linalg.norm(pat, axis=1).reshape(-1,1))
pat = normalize(pat, norm='l2')

gram_kernel = np.power(np.inner(pat, pat), powers)

# Gram kernel
m= 12  ## the number of prototype
selected, obj_list= select_greedy_protos(gram_kernel, m)

# color = cm.rainbow(np.linspace(0,1,m))[::-1]
# classified = np.argmax(gram_kernel[:, selected], axis=1)
#
# (unique, counts) = np.unique(classified, return_counts=True)


color = cm.rainbow(np.linspace(0,1,m))[::-1]
classified = np.argmax(gram_kernel[:, selected], axis=1)
yrange=(-3.5,3.5)
protos = subsequences[selected[:m]]

row = int((m-1)/12)+1
column = 16
# fig = plt.figure(figsize=(25,3*row))
figs, axes = plt.subplots(4, 3, figsize=(7, row*6))
gs = gridspec.GridSpec(row,column)
gs.update(wspace=0, hspace=0)

for n in range(m):
    group_idx = [i for i,x in enumerate(classified) if x == n]
    members = subsequences[classified==n]
    # proto_std = members.std(axis=0)
    proto_mean = members.mean(axis=0)
    # ax = plt.subplot(gs[n])

    if n < 3:
       i = 0
       j = n % 3
    elif n < 6:
       i = 1
       j = n % 3
    elif n < 9: 
       i = 2
       j = n % 3
    elif n < 12: 
       i = 3
       j = n % 3
    axes[i, j].plot(proto_mean, color='black',alpha=0.6)
    axes[i, j].plot(protos[n], color =color[n],alpha=1,linewidth=5)
    axes[i, j].set_ylim(yrange)
    
    axes[i, j].set_xticks([])
    axes[i, j].set_yticks([])
    plt.setp(axes[i, j].spines.values(), linewidth=1.5)
    
plt.tight_layout()
plt.savefig(save_path + '/plots/Prototypes.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()

proto_idx = 0
display_n = 9

group_idx = [i for i,x in enumerate(classified) if x == proto_idx]
members = subsequences[classified==proto_idx]
proto_std = members.std(axis=0)
proto_mean = members.mean(axis=0)

plt.figure(figsize=(2,2))
plt.plot(protos[proto_idx], color =color[proto_idx],alpha=1,linewidth=5)
plt.ylim(-5,5)
plt.xticks([])
plt.yticks([])
plt.title("Prototype {}".format(proto_idx))
plt.show()

fig = plt.figure(figsize=(1.7*display_n, 2))
gs = gridspec.GridSpec(1,display_n)
gs.update(wspace=0, hspace=0)

indices =  np.random.choice(len(members), display_n, replace=False)
for t, member in enumerate(members[indices]):
    ax = plt.subplot(gs[t])
    ax.plot(member, linewidth=2.5, color='black')

    ax.set_ylim(yrange)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax.spines.values(), linewidth=2)
    
plt.suptitle("PTAPs in Prototype {}".format(proto_idx), y=1.1, fontsize=18)
plt.tight_layout()
plt.show()

data_indices = np.array([x['data_idx'] for x in TAP])
subsequences = np.array([x['pattern_ys'] for x in TAP])
subseq_x = np.array([x['pattern_xs'] for x in TAP])
features = np.array([x['features'] for x in TAP])

np.argmax(y_train, axis=0)

tar_class = 0
tar_set = np.where(y_train==tar_class)[0]
print(tar_set)

####################### This part is to choose patterns randomly without overlapping. ############# 
idx = np.random.choice(tar_set, 1)[0]

# idx = 2795
pattern_list = np.where(data_indices==idx)[0]
pattern_list = pattern_list[np.argsort(-np.max(gram_kernel[pattern_list, :], axis=1))]

pattern_list2 = []
for i, p in enumerate(pattern_list):
    if i==0:
        pattern_list2.append(p)
    else:
        add = True
        for p2 in pattern_list2:
            if np.abs(subseq_x[p][0] - subseq_x[p2][0])<150:
                add = False
                break
        if add:
            pattern_list2.append(p)

########################################################################################################

fig = plt.figure(figsize=(8,3))

plt.title("Data {} | Class {}".format(idx, tar_class),fontsize=15)
plt.plot(trainx[idx].squeeze(), c='black')

# p_idx = pattern_list[0]
corres_p_list = []
for i, p_idx in  enumerate(pattern_list2):
    proto_idx = classified[p_idx]
    corres_p_list.append(proto_idx)
    
    members = subsequences[classified==proto_idx]
    proto_std = members.std(axis=0)*1.5
    proto_mean = members.mean(axis=0)
    
    plt.fill_between(subseq_x[p_idx, :], proto_mean-proto_std, proto_mean+proto_std, color=color[proto_idx], alpha=0.2)
    plt.plot(subseq_x[p_idx, :], protos[proto_idx], c=color[proto_idx], linewidth=5, alpha=0.75)

plt.xticks([])
plt.savefig(save_path + f'/plots/Prototypeswithts_{idx}.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.ylim(-3,3)

display_n = 5
column = 1
row = int((display_n-1)/column)+1

fig = plt.figure(figsize=(5,row*1.5))
gs = gridspec.GridSpec(row, column)
gs.update(wspace=0, hspace=0)

tar_class = 0
tar_set = np.where(y_train==tar_class)[0]
idxes = np.random.choice(tar_set, display_n)
# samples = np.random.choice(len(subsequences), display_n, replace=False)
print(idxes)
for i, n in enumerate(idxes):
    ax = plt.subplot(gs[i])
    ax.plot(trainx[i].squeeze(), c='black', linewidth=1.5)
    ax.set_ylim(-3,3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax.spines.values(), linewidth=2)
plt.tight_layout()
plt.savefig(save_path + '/plots/inputsequences.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()

tar_class = 0
tar_set = np.where(y_train==tar_class)[0]
print(tar_set)

####################### This part is to choose patterns randomly without overlapping. ############# 
idx = np.random.choice(tar_set, 1)[0]

# idx = 2795
pattern_list = np.where(data_indices==idx)[0]
pattern_list = pattern_list[np.argsort(-np.max(gram_kernel[pattern_list, :], axis=1))]

pattern_list2 = []
for i, p in enumerate(pattern_list):
    if i==0:
        pattern_list2.append(p)
    else:
        add = True
        for p2 in pattern_list2:
            if np.abs(subseq_x[p][0] - subseq_x[p2][0])<150:
                add = False
                break
        if add:
            pattern_list2.append(p)

########################################################################################################

fig = plt.figure(figsize=(8,3))

plt.title("Data {} | Class {}".format(idx, tar_class),fontsize=15)
plt.plot(trainx[idx].squeeze(), c='black')

# p_idx = pattern_list[0]
corres_p_list = []
for i, p_idx in  enumerate(pattern_list2):
    proto_idx = classified[p_idx]
    corres_p_list.append(proto_idx)
    
    members = subsequences[classified==proto_idx]
    proto_std = members.std(axis=0)*1.5
    proto_mean = members.mean(axis=0)
    
    plt.fill_between(subseq_x[p_idx, :], proto_mean-proto_std, proto_mean+proto_std, color=color[proto_idx], alpha=0.2)
    plt.plot(subseq_x[p_idx, :], protos[proto_idx], c=color[proto_idx], linewidth=5, alpha=0.75)

plt.xticks([])
plt.savefig(save_path + f'/plots/Prototypeswithts_{idx}.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.ylim(-3,3)
