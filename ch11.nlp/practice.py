# %%
# 합성곱 신경망: 이미지 처리에 탁월한성능 ~> 텍스터 처리 위한 시도
# 커널 : 이미지 추출을 위한 3x3, 5x5 합성곱 (훑어가는 녀석)
# stride: 이동 범위 지정  
# 합성곱에 bais(편향)은 모든 원소에 더해짐

# Oh,Ow = floor((lh - kh + 2p)/s +1)
# floor((입력높이 - 커널높이 + 2x패딩) / 스트라이프 + 1)

# 이미지의 경우 3개의 채널이 추가됨
# 즉 1채널 -> 높이x너비 * red
# other 채널 -> 높이x너비 * green,blue
# 특성맵 total

# 가중치 매개변수: Ki * Ko * Ci * Co
# 합성곱 정의
# 합성곱 연산이란 커널(kernel) 또는 필터(filter) 라는 n × m 크기의 행렬로 높이(height) × 너비(width) 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서 n × m 크기의 겹쳐지는 부분의 각 이미지와 커널의 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 것을 말합니다. 이때, 이미지의 가장 왼쪽 위부터 가장 오른쪽 아래까지 순차적으로 훑습니다
# model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
"""
num_filters: 커널 개수
kenel_size: 커널 크기
padding: 패딩 방법
 valid: 패딩 x
 same: 입력과 동일한 차원가지도록 제로 패딩 추가
activation: 활성화 함수
"""
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
# %%
embedding_dim = 256 # 임베팅 벡터 차원
dropout_ratio = 0.3 # 드롭아웃 비율
num_filters = 256 # 커널의 수
kernel_size = 3 # 커널 사이즈
hidden_units = 128 # 뉴런 수

vocab_size = 10000

model = Sequential()
model.add(Embedding(vocab_size,embedding_dim))
model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters,kernel_size,padding='valid',activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_units,activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(1,activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[es, mc])

# %%
# Intent Classification

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/11.%201D%20CNN%20Text%20Classification/dataset/intent_train_data.csv", filename="intent_train_data.csv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/11.%201D%20CNN%20Text%20Classification/dataset/intent_test_data.csv", filename="intent_test_data.csv")

train_data = pd.read_csv('intent_train_data.csv')
test_data = pd.read_csv('intent_test_data.csv')
# %%
train_data
# %%
test_data
# %%
intent_train = train_data['intent'].tolist()
label_train = train_data['label'].tolist()
intent_test = test_data['intent'].tolist()
label_test = test_data['label'].tolist()

print('훈련용 문장의 수 :', len(intent_train))
print('훈련용 레이블의 수 :', len(label_train))
print('테스트용 문장의 수 :', len(intent_test))
print('테스트용 레이블의 수 :', len(label_test))
# %%
train_data['label'].value_counts().plot(kind='bar')
# %%
# 레이블 인코딩= 레이블에 고유한 정수 부여
idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(label_train)

label_train = idx_encode.transform(label_train)
label_test = idx_encode.transform(label_test)

label_idx = dict(zip(list(idx_encode.classes_),idx_encode.transform(list(idx_encode.classes_))))
print('레이블 정수의 맵핑 관계 :',label_idx)
# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(intent_train)
sequences = tokenizer.texts_to_sequences(intent_train)
sequences[:5]
# %%
word_index = tokenizer.word_index
vocab_size = len(word_index)+1
print('Vocabluary 크기 :',vocab_size)
# %%
print('문장의 최대 길이 :',max(len(l) for l in sequences))
print('문장의 평균 길이 :',sum(map(len, sequences))/len(sequences))
plt.hist([len(s) for s in sequences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# %%
"""
라벨에 대해선 원-핫 인코딩을 진행한다.
이유는 분류 classificatino 으로 명확하게 떨어지기 때문
"""
max_len = 35
intent_train = pad_sequences(sequences,maxlen = max_len)
label_train = to_categorical(np.asanyarray(label_train))
print('훈련 데이터의 크기(shape):', intent_train.shape)
print('훈련 데이터 레이블의 크기(shape):', label_train.shape)
# %%
indices = np.arange(intent_train.shape[0])
np.random.shuffle(indices)
print('랜덤 시퀀스 :',indices)
# %%
intent_train = intent_train[indices]
label_train = label_train[indices]

# 10%만 검증 데이터로 활용
n_of_val = int(0.1 * intent_train.shape[0])
print('검증 데이터의 개수 :',n_of_val)
# %%
X_train = intent_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = intent_train[-n_of_val:]
y_val = label_train[-n_of_val:]
X_test = intent_test
y_test = label_test
# %%
print('훈련 데이터의 크기(shape):', X_train.shape)
print('검증 데이터의 크기(shape):', X_val.shape)
print('훈련 데이터 레이블의 크기(shape):', y_train.shape)
print('검증 데이터 레이블의 크기(shape):', y_val.shape)
print('테스트 데이터의 개수 :', len(X_test))
print('테스트 데이터 레이블의 개수 :', len(y_test))
# %%
from pathlib import Path
glove_path = Path(__file__).resolve().parent.parent / 'glove.6B.100d.txt'
embedding_dict = dict()
f = open(glove_path,encoding='utf-8')
# %%
count = 0
for line in f:
    if count >7:
        break
    word_vector = line.split()
    word = word_vector[0]
    print(word)
    print(word_vector[1])
    count+=1
# %%
# 확인결과 100차원이다(즉 :1 -> 단어, 1: -> 워드 임베디드)
for line in f:
    word_vector = line.split()
    word = word_vector[0] # 어떤 글자인가, word_vector[1]: 단어 벡터
    word_vector_arr = np.asarray(word_vector[1:],dtype='float32')
    embedding_dict[word] = word_vector_arr
f.close()
print('%s개의 Embedding vector가 있습니다.' % len(embedding_dict))
# %%
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
print('임베딩 테이블의 크기(shape) :',np.shape(embedding_matrix))
# %%
for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# %%
# 1D CNN 의도 본류
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate

kernel_size = [2,3,5]
num_filters = 512
dropout_ratio = 0.5

model_input = Input(shape=(max_len,))
output = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                      input_length=max_len, trainable=False)(model_input)

conv_blocks = []

for size in kernel_size:
    conv = Conv1D(filters=num_filters,
                  kernel_size=size,
                  padding="valid",
                  activation="relu",
                  strides=1)(output) #층(설정)(입력) → "이 층에 output을 통과시킨 결과를 conv에 저장"
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
output = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
output = Dropout(dropout_ratio)(output)
model_output = Dense(len(label_idx), activation='softmax')(output)
model = Model(model_input, model_output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# %%
history = model.fit(X_train, y_train,
          batch_size=64,
          epochs=10,
          validation_data=(X_val, y_val))