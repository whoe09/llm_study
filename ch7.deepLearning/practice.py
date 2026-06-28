## 딥러닝 ? 머신 러닝의 특정한 분야 + 인공 신경망의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식

# 퍼셉트론
## 가중치(w)가 클수록 중요한 의미
## 각 입력값 * 가중치의 곱 >= 임계치 경우, 출력신호를 1로, 아닌 경우 0
# %%
# input-3dim / output-2dim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(2,input_dim=3,activation='softmax'))

# %%
model.summary()
# %%
# 코드 구현
model = Sequential()
# 8개 출력 + 4개 입력
model.add(Dense(8,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))

# 손실함수
## 간단테스트: msc
## 대부분 분류 문제: binaray_crossentropy
## 다중 클래스 분류: catagorical_crossentropy


# 경사 하강법
## 배치 경사 하강법: 전체 set을 보고 매개변수를 업데이트 한다.
## SGD: 랜덤으로(1개) 데이터를 계산하면서 진행
## Mini-batch: n개를 지정하여 학습진행 ex) model.fit(X_train,y_train,batch_size=128)

## epoch: 전체 데이터 set을 돌린횟수
## batch size: 몇개로 끊어서 돌릴지
## iteration (step): batch_size기준 몇번 돌지

# 역전파 <-- 어려움..
# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
train_text = "The earth is an awesome place live"

# 단어 집합 생성
tokenizer.fit_on_texts([train_text])
# 정수 인코딩
sub_text = "The earth is an great place live"
sequences = tokenizer.texts_to_sequences([sub_text])[0]

print("정수 인코딩 : ",sequences)
print("단어 집합 : ",tokenizer.word_index)
# %%
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='pre')


## linear : 디폴트 값으로 별도 활성화 함수 없이 입력 뉴런과 가중치의 계산 결과 그대로 출력.
## sigmoid : 이진 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
## softmax : 셋 이상의 선택지 중 하나를 택하는 다중 클래스 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
## relu : 은닉층에 주로 사용되는 활성화 함수.

# %%
# keras api 활용해보기
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 10개 입력
inputs = Input(shape=(10,))
hidden1 = Dense(64,activation='relu')(inputs)
hidden2 = Dense(64,activation='relu')(hidden1)
ouput = Dense(1,activation='sigmoid')(hidden2)
model = Model(inputs=inputs,ouput=ouput)

# %%
# 다중 입력
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

x = Dense(16, activation="relu")(inputA)
x = Dense(8, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(8, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# 두개의 인공 신경망의 출력을 연결(concatenate)
result = concatenate([x.output, y.output])

z = Dense(2, activation="relu")(result)
z = Dense(1, activation="linear")(z)

model = Model(inputs=[x.input, y.input], outputs=z)

# %%
# MLP = 다층 퍼셉트론 사용해보기
texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)
# %%
# 'count' -> index 0번이 추가됨, 등장 횟수
print(tokenizer.texts_to_matrix(texts, mode = 'count')) # texts_to_m
# %%
# 'binary' -> 존재여부에 따라 0/1
print(tokenizer.texts_to_matrix(texts, mode = 'binary'))
# %%
# 'tfidf' -> TF-IDF 행렬
print(tokenizer.texts_to_matrix(texts, mode = 'tfidf').round(2))
# %%
# 'freq' -> 
print(tokenizer.texts_to_matrix(texts, mode = 'freq').round(2))
# %%
