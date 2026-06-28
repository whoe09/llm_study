# 텍스트 분류 실습 = 지도 학습
# model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))
# hidden_units: RNN의 출력 크기(은닉 상태의 크기)
# timestemps = 시점의 수 = 각 문서에서의 단어 수
# input_dim = 입력 크기 = 임베딩 벡터 차원

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/10.%20RNN%20Text%20Classification/dataset/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin1')
print('총 샘플의 수 :',len(data))

# %%
data[:5]
# %%
# 필요없는 행 자르기
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data[:5]
# %%
data.info()

print('결측 확인 :',data.isnull().values.any())

# %%
# 중복값 확인
print('v2 유니크값 :',data['v2'].unique())
# %%
# 중복 제거 / drop_duplicates
# inplace = 원래 데이터 변형 할것인가
data.drop_duplicates(subset=['v2'],inplace=True)
print('총 샘플 수 :',len(data))

# %%
data['v1'].value_counts().plot(kind = 'bar')

# %%
print('정상 메일과 스팸 메일의 개수')
print(data.groupby('v1').size().reset_index(name='count'))

# %%
X_data = data['v2']
y_data = data['v1']
print('메일 본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))

# %%
# stratify: 레이블 분포를 고르게 분포(어떤놈을 기준으로 할지 인자를 주는것)
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2,random_state=0,stratify=y_data)
# %%
print('--------훈련 데이터의 비율-----------')
print(f'정상 메일 = {round(y_train.value_counts()[0]/len(y_train) * 100,3)}%')
print(f'스팸 메일 = {round(y_train.value_counts()[1]/len(y_train) * 100,3)}%')
print('--------테스트 데이터의 비율-----------')
print(f'정상 메일 = {round(y_test.value_counts()[0]/len(y_test) * 100,3)}%')
print(f'스팸 메일 = {round(y_test.value_counts()[1]/len(y_test) * 100,3)}%')

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)
print(X_train_encoded[:5])

# %%
# 단어 유추
word_to_index = tokenizer.word_index
print(word_to_index)

# %%
# 한줄만 읽어보기
print('1번 문장 정수 인코딩')
print(X_train_encoded[0])
index_to_word = {index: word for word, index in word_to_index.items()}
decoded = [index_to_word[i] for i in X_train_encoded[0]]
print('정수→단어 복원 :', decoded)

# %%
# 비율 확인
threshold = 2
total_cnt = len(word_to_index)
rare_cnt = 0 # 등장 빈도수가 threshold 보다 작은 단어 개수 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합


for key, value in tokenizer.word_counts.items():
    total_freq += value

    if (value < threshold):
        rare_cnt+=1
        rare_freq+=value
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 빈도수가 n인 단어 제외
# token = Tokenizer(num_words = total_cnt=rare_cnt+1)

# %%
vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))

print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))
plt.hist([len(sample) for sample in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# %%
max_len = 189
X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len)

# %%
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size,embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_padded, y_train, epochs=4, batch_size=64, validation_split=0.2)

X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test_padded, y_test)[1]))