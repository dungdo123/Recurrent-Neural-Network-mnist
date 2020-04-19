import pandas as pd
from collections import deque
import random
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import time
import keras

SEQ_LEN = 60
FUTURE_PERIDOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocessing_df(df):
    df = df.drop("future",1)

    for col in df.columns:
        if col != "target":
            df[col]= df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]])
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    Y = []
    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)

    return np.array(X),Y

main_df = pd.DataFrame()

ratios = ["BTC-USD","LTC-USD","BCH-USD","ETH-USD"]
for ratio in ratios:
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time','low','high','open','close','volume'])

    df.rename(columns={"close":f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace= True)

    df.set_index("time",inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)
#print(main_df.head())

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIDOD_PREDICT)
main_df['target'] = list(map(classify,main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index <= last_5pct)]

train_x, train_y = preprocessing_df(main_df)
validation_x, validation_y = preprocessing_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

#compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
# tensorboard = TensorBoard(log_dir="log/{}".format(NAME))
# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    # callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

