# %%
import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return np.array(x > 0, dtype=np.double)
x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = step(x)
plt.title('Step Function')
plt.plot(x,y)
plt.show()
# %%
def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# %%
# 주로 은닉층에서 사용(양수가 나오면 입력값을 그대로 반환, 음수의 경우 아주 작은 수 반환)
a = 0.1

def leaky_relu(x):
    return np.maximum(a*x, x)

x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()

# %%
# 세 가지 이상의 선택지 중 하나 고르는 MultiClass Classification 문제 사용
a = 0.1
x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.exp(x) / np.sum(np.exp(x))

plt.plot(x, y)
plt.title('Softmax Function')
plt.show()

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 입력 3개 -> 출력 2개
model = Sequential()
model.add(Dense(2,input_dim=3,activation='softmax'))
# %%
model.summary()
# 3개 입력x2 = 6개의 가중치(w) + 2개편향(b) = 출력값에 대한 편향

# %%
model = Sequential()

model.add(Dense(8,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,input_dim=8,activation='softmax'))
# %%
model.summary()
# dense_1 = 4*8 + 8 = 40
# dense_2 = 8*8 + 8 = 72
# dense_3 = 8*3 + 3 = 27
# total = 40 + 72 + 27 = 139

# %%
# MSM: MEan Squared Error / 평균 제곱 오차 
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
# %%
# Binary Cross-Entropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# %%
