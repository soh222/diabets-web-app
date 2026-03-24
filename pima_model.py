# 파이썬 3.5 이상이 필요합니다
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn 0.20 이상이 필요합니다
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow 2.0 이상이 필요합니다
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import os
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 이 노트북의 출력을 실행마다 일정하게 유지하기 위해 설정합니다
np.random.seed(42)
tf.random.set_seed(42)

# 예쁜 그래프를 그리기 위한 설정입니다
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 피마 인디언 당뇨병 데이터셋을 포함한 파일을 읽어옵니다
data = pd.read_csv('./diabetes.csv', sep=',')

print("\ndata.head(): \n",
      data.head())

# 데이터셋의 컬럼들을 설명합니다
data.describe()

# 데이터셋에 결측치(null)가 있는지 확인합니다
data.info()

print("\n\nStep 2 - 모델 구축을 위한 데이터 준비")
# 가져온 데이터에서 X와 y를 추출합니다
X = data.values[:, 0:8]
y = data.values[:, 8]

# MinMaxScaler를 사용하여 스케일러 객체를 맞춥니다(fit)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# 테스트 세트를 훈련 세트와 테스트 세트로 분할합니다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n\nStep 3 - 모델 생성 및 훈련")
# 모델 생성
inputs = keras.Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = keras.Model(inputs, output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# 에포크에 따른 손실(loss)과 정확도(accuracy)의 이력을 요약합니다
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color=color)
ax1.plot(history.history['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() # 동일한 x축을 공유하는 두 번째 축을 생성합니다

color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color) # x축 라벨은 이미 ax1에서 처리했습니다
ax2.plot(history.history['accuracy'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() # 이렇게 하지 않으면 오른쪽 y축 라벨이 약간 잘릴 수 있습니다
plt.show()

X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))

print("\nExporting SavedModels: ")

# Keras 모델 저장
model.save('pima_model.keras')

# 모델 로드
model = keras.models.load_model('pima_model.keras')

# 모델 평가
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))