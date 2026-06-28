#%%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 숫자 10부터 1

model = Sequential()
model.add(Dense(1,input_dim=1,activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=200)
plt.plot(x, model.predict(x), 'b', x,y, 'k.')
# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]]) 
y = np.array([73, 82 ,72, 57, 34])
model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))

sgd = optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=2000)

# %%
X_test = np.array([[20,99,10], [40,50,20]])
print(model.predict(X_test))

# %%
import tensorflow as tf

w = tf.Variable(2.)
def f(w):
    y = w**2
    z = 2*y +5
    return z

with tf.GradientTape() as tape:
    z = f(w)
gradients = tape.gradient(z,[w])
print(gradients)

# Vairable() : 메모리 내에서 텐서의 데티어 유지+갱신
w = tf.Variable(4.0)
b = tf.Variable(1.0)

@tf.function
def hypotheses(x):
    return w*x+b;

x_test = [3.5,5,5.5,6]
print(hypotheses(x_test).numpy())

@tf.function
def mse_loss(y_pred,y):
    return tf.reduce_mean(tf.square(y_pred-y))

x = [1,2,3,4,5,6,7,8,9]
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]


optimizer = tf.optimizers.SGD(0.01)


for i in range(301):
    with tf.GradientTape() as tape: #tf.GradientTape: 자동 미분
        y_pred = hypotheses(x)
        cost = mse_loss(y_pred,y)
    gradients = tape.gradient(cost,[w,b])
    optimizer.apply_gradients(zip(gradients, [w, b]))
    if i % 10 == 0:
        print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))


# %%
x_test = [3.5,5,5.5,6]
print(hypotheses(x_test).numpy())
# %%
def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # w의 값이 0.5일때
plt.plot(x, y2, 'g') # w의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # w의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
# %%
x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 숫자 10부터 1
model = Sequential()
model.add(Dense(1,input_dim=1,activation='sigmoid'))

sgd = optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=200)
# %%
plt.plot(x, model.predict(x), 'b', x,y, 'k.')
# %%
X = np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]]) 
y = np.array([73, 82 ,72, 57, 34]) # 최종 성적

model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))
sgd = optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=2000)

# %%
d = np.array(5)
print('텐서 차원 :',d.ndim)
print('텐서 크기: ',d.shape)
# %%
d = np.array([
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]]
            ])
# 5개 묶음이 3쌍있는 2개의 박스
print('텐서의 차원 :',d.ndim)
print('텐서의 크기(shape) :',d.shape)
# %%
# softmax 실습
model = Sequential()
model.add(Dense(3,input_dim=4,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))
