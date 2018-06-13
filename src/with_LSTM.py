#!/usr/bin/env python3
# coding: utf-8
# File: with_LSTM.py
# Author: lxw
# Date: 6/6/18 5:26 PM

import numpy as np
import pandas as pd
import pickle
import time

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.models import load_model
from keras.utils import np_utils

# from src.preprocessing import gen_train_val_test_data
from preprocessing import gen_train_val_test_data


def model_build(input_shape, num_classes=5):
    """
    :param input_shape: 
    :return: 
    """
    model = Sequential()

    # The way Keras LSTM layers work is by taking in a numpy array of 3 dimensions (N, W, F) where N is the
    # number of training sequences, W is the sequence length and F is the number of features of each sequence.
    model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True, name="lstm1"))
    # model.add(GRU(units=64, input_shape=input_shape, return_sequences=True, name="gru1"))
    model.add(Dropout(0.25, name="dropout2"))

    model.add(LSTM(units=128, return_sequences=False, name="lstm3"))
    # model.add(GRU(units=128, return_sequences=False, name="gru3"))
    model.add(Dropout(0.25, name="dropout4"))

    """
    model.add(LSTM(units=layers[2], return_sequences=False, name="lstm7"))
    # model.add(GRU(units=layers[2], return_sequences=False, name="gru7"))
    model.add(Dropout(0.25, name="dropout8"))

    model.add(LSTM(units=layers[3], return_sequences=True, name="lstm9"))
    model.add(Dropout(0.25, name="dropout10"))

    model.add(LSTM(units=layers[4], return_sequences=True, name="lstm11"))
    model.add(Dropout(0.25, name="dropout12"))

    model.add(LSTM(units=layers[5], return_sequences=False, name="lstm13"))
    model.add(Dropout(0.25, name="dropout14"))
    """

    model.add(Dense(units=num_classes, activation="softmax", name="dense5"))

    start = time.time()
    # optimizer="rmsprop". This optimizer is usually a good choice for Recurrent Neural Networks.
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("> Compilation Time: ", time.time() - start)

    return model


def model_train_val(X_train, X_val, y_train, y_val):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    print("X_train.shape:{0}\nX_val.shape:{1}\n".format(X_train.shape, X_val.shape))

    model = model_build(input_shape=(X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)

    BATCH_SIZE = 1024
    EPOCHS = 300
    # NOTE: It's said and I do think monitor="val_loss" is better than "val_acc".
    # Reference: [Should we watch val_loss or val_acc in callbacks?](https://github.com/raghakot/keras-resnet/issues/41)
    lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, factor=0.8, min_lr=0.00001)
    # hist_obj = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    hist_obj = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                         validation_data=(X_val, y_val), callbacks=[lr_reduction, early_stopping])
    with open("../data/output/history.pkl", "wb") as f:
        pickle.dump(hist_obj.history, f)

    model.save("../data/output/models/lstm.model")


def plot_hist():
    import matplotlib.pyplot as plt

    history = None
    # DEBUG
    # with open("../data/output/history_295_1024.pkl", "rb") as f:
    # with open("../data/output/history_50_512.pkl", "rb") as f:
    # with open("../data/output/history_69_128.pkl", "rb") as f:
    # with open("../data/output/history_74_64.pkl", "rb") as f:
    # with open("../data/output/history_16_256.pkl", "rb") as f:
    # with open("../data/output/history_50_32.pkl", "rb") as f:
    # with open("../data/output/history_65_16.pkl", "rb") as f:
    # with open("../data/output/history_26_128.pkl", "rb") as f:
    # with open("../data/output/history_88_512.pkl", "rb") as f:
    with open("../data/output/history_111_1024.pkl", "rb") as f:
        history = pickle.load(f)
    if not history:
        return
    # 绘制训练集和验证集的曲线
    plt.plot(history["acc"], label="Training Accuracy", color="green", linewidth=2)
    plt.plot(history["loss"], label="Training Loss", color="red", linewidth=1)
    plt.plot(history["val_acc"], label="Validation Accuracy", color="purple", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", color="blue", linewidth=1)
    plt.grid(True)  # 设置网格形式
    plt.xlabel("epoch")
    plt.ylabel("acc-loss")  # 给x, y轴加注释
    plt.legend(loc="upper right")  # 设置图例显示位置
    plt.show()


def model_predict(model, X_test, X_test_id, X_val, y_val):
    # Generate predicted result.
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    print("X_test.shape:{0}\nX_val.shape:{1}\n".format(X_test.shape, X_val.shape))
    predicted = model.predict(X_test)  # predicted.shape: (, )
    """
    # print(predicted[:10])  # OK
    [[ 0.17622797  0.17507555  0.27694944  0.19135155  0.18039554]
     ...
     [ 0.17644432  0.17531542  0.27615064  0.19144376  0.18064587]
     [ 0.17644432  0.17531542  0.27615064  0.19144376  0.18064587]
     [ 0.17644432  0.17531544  0.27615064  0.19144376  0.18064587]]
    """
    # 把categorical数据转为numeric值，得到分类结果
    predicted = np.argmax(predicted, axis=1)
    """
    np.savetxt("../data/output/lstm_submission.csv", np.c_[range(1, len(X_test) + 1), predicted], delimiter=",",
               header="PhraseId,Sentiment", comments="", fmt="%d")
    """
    predicted = pd.Series(predicted, name="Sentiment")
    submission = pd.concat([X_test_id, predicted], axis=1)
    submission.to_csv("../data/output/lstm_submission.csv", index=False)

    # Model Evaluation
    print("model.metrics:{0}, model.metrics_names:{1}".format(model.metrics, model.metrics_names))
    scores = model.evaluate(X_val, y_val)
    loss, accuracy = scores[0], scores[1] * 100
    print("Loss: {0:.2f}, Model Accuracy: {1:.2f}%".format(loss, accuracy))


if __name__ == "__main__":
    """
    X_train, X_val, X_test, X_test_id, y_train, y_val = gen_train_val_test_data()
    print("X_train.shape:{0}\nX_val.shape:{1}\nX_test.shape:{2}\nX_test_id.shape:{3}\n"
          "y_train.shape:{4}\ny_val.shape:{5}\n".format(X_train.shape, X_val.shape, X_test.shape,
                                                         X_test_id.shape, y_train.shape, y_val.shape))

    # model_train_val(X_train, X_val, y_train, y_val)
    """

    plot_hist()

    """
    model = load_model("../data/output/models/lstm_50.model")   # DEBUG
    model_predict(model, X_test, X_test_id, X_val, y_val)
    """

