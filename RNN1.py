import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.datasets import mnist
import keras
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255.0
y_train = y_train/255.0



model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))

model.save("LSTM_mnist.model")

