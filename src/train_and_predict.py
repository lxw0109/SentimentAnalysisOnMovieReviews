#!/usr/bin/env python3
# coding: utf-8
# File: train_and_predict.py
# Author: lxw
# Date: 6/6/18 1:46 PM

import json
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import classification_report

# from src.preprocessing import gen_train_eval_data
from preprocessing import gen_train_eval_data


def fetch_data():
    """
    :return: 
    """
    train_df = pd.read_csv("../data/output/train_vector.csv", sep="\t")  # (156060, 2)
    X_train, X_eval, y_train, y_eval = gen_train_eval_data(train_df)

    test_df = pd.read_csv("../data/output/test_vector.csv", sep="\t")  # (156060, 2)
    X_test = test_df["Phrase_vec"]  # <Series>. shape: (,)
    X_test = np.array([json.loads(vec) for vec in X_test])
    X_test_id = test_df["PhraseId"]  # <Series>. shape: (,)
    X_test_id = np.array(X_test_id)
    return X_train, X_eval, X_test, X_test_id, y_train, y_eval


def train_eval_test(X_train, X_eval, X_test, X_test_id, y_train, y_eval):
    '''
    # 1. [NO]LR: LR算法的优点是可以给出数据所在类别的概率
    model = linear_model.LogisticRegression(C=1e5)
    """
    C: default: 1.0
    Inverse of regularization strength; must be a positive float. Like in support vector machines,
    smaller values specify stronger regularization.
    """

    # 2. [NO]NB: 也是著名的机器学习算法, 该方法的任务是还原训练样本数据的分布密度, 其在多分类中有很好的效果
    from sklearn import naive_bayes
    model = naive_bayes.GaussianNB()  # 高斯贝叶斯

    # 3. [OK_but_too_slow]KNN:
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()  # 非常慢，感觉没法用(跑了半个多小时没反应)

    # 4. [OK]决策树: 分类与回归树(Classification and Regression Trees, CART)算法常用于特征含有类别信息
    # 的分类或者回归问题，这种方法非常适用于多分类情况
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()

    # 5. [NO]SVM: SVM是非常流行的机器学习算法，主要用于分类问题，
    # 如同逻辑回归问题，它可以使用一对多的方法进行多类别的分类
    from sklearn.svm import SVC
    model = SVC()
    '''

    # 6. MLP: 多层感知器(神经网络)
    from sklearn.neural_network import MLPClassifier
    # model = MLPClassifier(activation="relu", solver="adam", alpha=0.0001)
    # model = MLPClassifier(activation="identity", solver="adam", alpha=0.0001)
    # model = MLPClassifier(activation="logistic", solver="adam", alpha=0.0001)
    model = MLPClassifier(activation="tanh", solver="adam", alpha=0.0001)

    model.fit(X_train, y_train)
    y_eval_pred = model.predict(X_eval)
    print(classification_report(y_eval, y_eval_pred))  # y_true, y_pred
    Z = model.predict(X_test)
    print(Z)


if __name__ == "__main__":
    X_train, X_eval, X_test, X_test_id, y_train, y_eval = fetch_data()
    print("X_train.shape:{0}\nX_eval.shape:{1}\nX_test.shape:{2}\nX_test_id.shape:{3}\ny_train.shape:{4}\ny_eval.shape:{5}\n".\
        format(X_train.shape, X_eval.shape, X_test.shape, X_test_id.shape, y_train.shape, y_eval.shape))
    train_eval_test(X_train, X_eval, X_test, X_test_id, y_train, y_eval)

