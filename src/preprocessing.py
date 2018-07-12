#!/usr/bin/env python3
# coding: utf-8
# File: preprocessing.py
# Author: lxw
# Date: 5/14/18 10:53 PM

import collections
import json
import numpy as np
import nltk
import pandas as pd
import time

from gensim.models import KeyedVectors
from gensim.models import word2vec
from keras.preprocessing import sequence
from keras.utils import np_utils
# from pyfasttext import FastText
from sklearn.model_selection import train_test_split


class MyEncoder(json.JSONEncoder):  # TO avoid: "TypeError: Object of type 'int64' is not JSON serializable"
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def fetch_data_df(train_path, test_path, sep="\t"):
    """
    :param train_path: path of train data.
    :param test_path:  path of test data.
    :param sep: 
    :return: return train_df and test_df **WITHOUT Normalization**.
    """
    train_df, test_df = None, None
    if train_path:
        train_df = pd.read_csv(train_path, sep=sep)  # (156060, 4)
    if test_path:
        test_df = pd.read_csv(test_path, sep=sep)  # (66292, 3)
    # print(train_df.describe())
    # print(test_df.describe())
    return train_df, test_df


def data_analysis(train_df, test_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white", context="notebook", palette="deep")

    y_train = train_df["Sentiment"]
    X_train = train_df.drop(labels=["Sentiment"], axis=1)

    # free some space
    del train_df

    # 1. 样本数据的均衡性统计(各个label数据是否均匀分布)
    sns.countplot(y_train)
    plt.show()
    print(y_train.value_counts())  # Is this an imbalanced dataset? NO
    """
    2    79582
    3    32927
    1    27273
    4     9206
    0     7072
    """

    # 2. Check for null and missing values
    # print(pd.DataFrame([1, 2, 3, np.nan, 1, 2, 3, -1, 3, 2, 1, 3, 2, np.nan, 3, 2, 1]).isnull().any())
    # print(pd.DataFrame([1, 2, 3, np.nan, 1, 2, 3, -1, 3, 2, 1, 3, 2, np.nan, 3, 2, 1]).isnull().any().describe())
    # 2.1 Method 1
    print(X_train.isnull().any().describe())  # no misssing values.
    print(test_df.isnull().any().describe())  # no misssing values.

    # 2.2 Method 2
    print(X_train.info())

    # fillna() if missing values occur.


def rm_stopwords(train_df, test_df):
    """
    分词 -> 去停用词
    生成文件"../data/output/train_wo_sw.csv" 和 "test_wo_sw.csv"
    :param train_df: 
    :param test_df: 
    :return: 
    """
    # 1. load stopwords.
    stop_words_list = open("../data/input/snownlp_en_stopwords.txt").readlines()
    stop_words_set = set()
    for word in stop_words_list:
        word = word.strip()
        if word:
            stop_words_set.add(word)

    # 2. process train_df
    phrase_series = train_df["Phrase"]  # <Series>. shape: (156060,)
    sentiment_series = train_df["Sentiment"]  # <Series>. shape: (156060,)

    f = open("../data/output/train_wo_sw.csv", "wb")
    f.write("Phrase\tSentiment\n".encode("utf-8"))  # NOTE: 不能以逗号分割，因为数据中有逗号分割的词，例如数字中的分隔符
    for ind, phrase in enumerate(phrase_series):
        word_list = phrase.split()
        word_wo_sw = []
        for word in word_list:
            if word not in stop_words_set and word != "":
                word_wo_sw.append(word)
        # if word_wo_sw:  # 空的也得写入文件, 后面预测时也会出现空的情况, 所以这里需要在训练集中也出现
        f.write("{0}\t{1}\n".format(" ".join(word_wo_sw), sentiment_series.iloc[ind]).encode("utf-8"))
    f.close()

    # 3. process test_df
    phrase_series = test_df["Phrase"]  # <Series>. shape: (156060,)
    phrase_id_series = test_df["PhraseId"]  # <Series>. shape: (156060,)
    f = open("../data/output/test_wo_sw.csv", "wb")
    f.write("PhraseId\tPhrase\n".encode("utf-8"))
    for ind, phrase in enumerate(phrase_series):
        word_list = phrase.split()
        word_wo_sw = []
        for word in word_list:
            if word not in stop_words_set and word != "":
                word_wo_sw.append(word)
        # if word_wo_sw:  # 空的也得写入文件, 后面还是要进行预测的
        f.write("{0}\t{1}\n".format(phrase_id_series.iloc[ind], " ".join(word_wo_sw)).encode("utf-8"))
    f.close()


def data2vec(train_df, test_df):
    """
    word2vec(phrase2vec), 并将结果写入文件output/train_vector.csv, output/test_vector.csv
    :param train_df: 
    :param test_df: 
    :return: 
    """
    # 1. 加载模型
    start_time = time.time()
    model = KeyedVectors.load_word2vec_format("../data/input/models/GoogleNews-vectors-negative300.bin", binary=True)
    # model = FastText("/home/lxw/IT/program/github/NLP-Experiments/fastText/data/lxw_model_cbow.bin")  # OK
    # model = KeyedVectors.load_word2vec_format("/home/lxw/IT/program/github/NLP-Experiments/word2vec/data/"
    #                                           "corpus.model.bin", binary=True)
    end_time = time.time()
    print("Loading Model Time Cost: {}".format(end_time - start_time))
    model_word_set = set(model.index2word)
    vec_size = model.vector_size
    # model.index2entity == model.index2word: True
    # print(model.similarity("good", "bad"))  # 0.7190051208276236

    # 2. 生成Phrase vector
    # Reference: [在python中如何用word2vec来计算句子的相似度](https://vimsky.com/article/3677.html)
    senti_series = train_df["Sentiment"]  # <Series>. shape: (156060,)
    phrase_series = train_df["Phrase"]  # <Series>. shape: (156060,)
    f = open("../data/output/train_vector_lower.csv", "wb")
    f.write("Phrase_vec\tSentiment\n".encode("utf-8"))  # NOTE:不能以逗号分割,因为数据中有逗号分割的词,如数字中的分隔符
    for ind, phrase in enumerate(phrase_series):
        phrase = str(phrase).lower()
        phrase_vec = np.zeros((vec_size,), dtype="float32")
        word_count = 0
        word_list = phrase.split()
        for word in word_list:
            if word in model_word_set:
                word_count += 1
                phrase_vec = np.add(phrase_vec, model[word])
        if word_count > 0:
            phrase_vec = np.divide(phrase_vec, word_count)
        f.write("{0}\t{1}\n".format(json.dumps(phrase_vec.tolist()), senti_series.iloc[ind]).encode("utf-8"))
    f.close()

    phrase_id_series = test_df["PhraseId"]  # <Series>. shape: (156060,)
    phrase_series = test_df["Phrase"]  # <Series>. shape: (156060,)
    f = open("../data/output/test_vector_lower.csv", "wb")
    f.write("PhraseId\tPhrase_vec\n".encode("utf-8"))  # NOTE: 不能以逗号分割，因为数据中有逗号分割的词，例如数字中的分隔符
    for ind, phrase in enumerate(phrase_series):
        phrase = str(phrase).lower()
        phrase_vec = np.zeros((vec_size,), dtype="float32")
        word_count = 0
        word_list = phrase.split()
        for word in word_list:
            if word in model_word_set:
                word_count += 1
                phrase_vec = np.add(phrase_vec, model[word])
        if word_count > 0:
            phrase_vec = np.divide(phrase_vec, word_count)
        f.write("{0}\t{1}\n".format(phrase_id_series.iloc[ind], json.dumps(phrase_vec.tolist())).encode("utf-8"))
    f.close()


def data2matrix(train_df, test_df):
    """
    matrix of phrase vector, 并将结果写入文件../data/output/train_matrix.csv, ../data/output/test_matrix.csv
    :param train_df: 
    :param test_df: 
    :return: max_phrase_length
    """
    # 1. 加载模型
    start_time = time.time()
    model = KeyedVectors.load_word2vec_format("../data/input/models/GoogleNews-vectors-negative300.bin", binary=True)
    # model = FastText("/home/lxw/IT/program/github/NLP-Experiments/fastText/data/lxw_model_cbow.bin")  # OK
    # model = KeyedVectors.load_word2vec_format("/home/lxw/IT/program/github/NLP-Experiments/word2vec/data/"
    #                                           "corpus.model.bin", binary=True)
    end_time = time.time()
    print("Loading Model Time Cost: {}".format(end_time - start_time))
    model_word_set = set(model.index2word)
    """
    with open("../data/output/word_set.json", "wb") as f:
        f.write(json.dumps(list(model_word_set)).encode("utf-8"))
    """
    # vec_size = model.vector_size
    # model.index2entity == model.index2word: True
    # print(model.similarity("good", "bad"))  # 0.7190051208276236

    # 2. 生成Phrase vector
    senti_series = train_df["Sentiment"]  # <Series>. shape: (156060,)
    phrase_series = train_df["Phrase"]  # <Series>. shape: (156060,)
    # f = open("../data/output/train_matrix_lower.csv", "wb")
    f = open("../data/output/train_matrix.csv", "wb")
    f.write("Phrase_vec\tSentiment\n".encode("utf-8"))  # NOTE:不能以逗号分割,因为数据中有逗号分割的词,如数字中的分隔符
    max_phrase_length = 0
    empty_statistics_train = {}
    for ind, phrase in enumerate(phrase_series):
        # phrase = str(phrase).lower()
        phrase = str(phrase)
        phrase_matrix = []  # list of list.
        phrase_length = 0
        word_list = phrase.split()
        for word in word_list:
            if word in model_word_set:
                phrase_length += 1
                phrase_matrix.append(model[word].tolist())  # type(model[word]): ndarray
        if phrase_length > max_phrase_length:
            max_phrase_length = phrase_length
        if phrase_length > 0:
            f.write(f"{json.dumps(phrase_matrix)}\t{senti_series.iloc[ind]}\n".encode("utf-8"))
        else:  # phrase_length == 0
            if senti_series.iloc[ind] in empty_statistics_train:
                empty_statistics_train[senti_series.iloc[ind]] += 1
            else:
                empty_statistics_train[senti_series.iloc[ind]] = 1
    f.close()
    print(f"empty_statistics_train: {empty_statistics_train}")
    empty_statistics_train = list(empty_statistics_train.items())
    empty_statistics_train = sorted(empty_statistics_train, key=lambda x: x[1], reverse=True)
    print(f"empty_statistics_train: {empty_statistics_train}")
    most_senti = empty_statistics_train[0][0]
    print(f"most_senti: {most_senti}")

    phrase_id_series = test_df["PhraseId"]  # <Series>. shape: (156060,)
    phrase_series = test_df["Phrase"]  # <Series>. shape: (156060,)
    # f = open("../data/output/test_matrix_lower.csv", "wb")
    f = open("../data/output/test_matrix.csv", "wb")
    empty_matrix_list_test = list()  # list of empty matrix, identified by phrase_id. 
    f.write("PhraseId\tPhrase_vec\n".encode("utf-8"))  # NOTE: 不能以逗号分割，因为数据中有逗号分割的词，例如数字中的分隔符
    for ind, phrase in enumerate(phrase_series):
        # phrase = str(phrase).lower()
        phrase = str(phrase)
        phrase_matrix = []  # list of list.
        phrase_length = 0
        word_list = phrase.split()
        for word in word_list:
            if word in model_word_set:
                phrase_length += 1
                phrase_matrix.append(model[word].tolist())  # type(model[word]): ndarray
        if phrase_length > max_phrase_length:
            max_phrase_length = phrase_length
        if phrase_length > 0:
            f.write(f"{phrase_id_series.iloc[ind]}\t{json.dumps(phrase_matrix)}\n".encode("utf-8"))
        else:
            # print("---NO: empty matrix in test.---")
            # print(phrase_id_series.iloc[ind])
            empty_matrix_list_test.append(phrase_id_series.iloc[ind])
            # f.write(f"{phrase_id_series.iloc[ind]}\t{most_senti}".encode("utf-8"))  # 这里统计训练集中label(senti)最多的填入(不用在这里填入, 直接在最后的submission.csv文件中填)

    f.close()

    print(f"max_phrase_length: {max_phrase_length}")
    fill_train_test_matrix(max_phrase_length)
    # 把空矩阵的phrase_id列表写入文件，将来往submission中填
    with open("../data/output/submissions/empty_matrix_list_test.txt", "wb") as f:
        f.write(f"{most_senti}\n".encode("utf-8"))
        f.write(json.dumps(empty_matrix_list_test, cls=MyEncoder).encode("utf-8"))  # "cls=MyEncoder", TO avoid: "TypeError: Object of type 'int64' is not JSON serializable"


def fill_train_test_matrix(max_phrase_length):
    """
    补齐"../data/output/train_matrix_lower.csv"和"../data/output/test_matrix_lower.csv"到最大短语长度(max_phrase_length)
    or
    补齐"../data/output/train_matrix.csv"和"../data/output/test_matrix.csv"到最大短语长度(max_phrase_length)
    :return: 
    """
    # TODO: 这儿感觉不要用pandas，否则内存压力太大？ pandas是一次性把所有的数据都读到内存中吗?感觉是, 待验证
    # 1. 补齐 "../data/output/train_matrix_lower.csv" or "../data/output/train_matrix.csv"
    # f1 = open("../data/output/train_matrix_lower_pad.csv", "wb")
    f1 = open("../data/output/train_matrix_pad.csv", "wb")
    # with open("../data/output/train_matrix_lower.csv") as f:
    with open("../data/output/train_matrix.csv") as f:
        f1.write(f.readline().encode("utf-8"))
        for line in f:
            line = line.strip()
            matrix, label = line.split("\t")
            matrix = json.loads(matrix)  # matrix: list of list
            # print(f"type(matrix): {type(matrix)}, type(matrix[0]): {type(matrix[0])}")
            # matrix  = np.array([json.loads(vec) for vec in matrix])
            length = len(matrix)
            assert length <= max_phrase_length
            # print(f"max_phrase_length: {max_phrase_length}, to be filled: {max_phrase_length-length}")
            matrix = np.pad(matrix, pad_width=((0, max_phrase_length-length), (0, 0)), mode="constant",
                            constant_values=-1)  # 参数中的matrix类型为list of list, 返回值的matrix是ndarray类型
            f1.write(f"{json.dumps(matrix.tolist())}\t{label}\n".encode("utf-8"))
    f1.close()

    # 2. 补齐 "../data/output/test_matrix_lower.csv" or "../data/output/test_matrix.csv"
    # f1 = open("../data/output/test_matrix_lower_pad.csv", "wb")
    f1 = open("../data/output/test_matrix_pad.csv", "wb")
    # with open("../data/output/test_matrix_lower.csv") as f:
    with open("../data/output/test_matrix.csv") as f:
        f1.write(f.readline().encode("utf-8"))
        for line in f:
            line = line.strip()
            phrase_id, matrix = line.split("\t")
            matrix = json.loads(matrix)  # matrix: list of list
            # matrix  = np.array([json.loads(vec) for vec in matrix])
            length = len(matrix)
            assert length <= max_phrase_length
            matrix = np.pad(matrix, pad_width=((0, max_phrase_length-length), (0, 0)), mode="constant",
                            constant_values=-1)  # 参数中的matrix类型为list of list, 返回值的matrix是ndarray类型
            f1.write(f"{phrase_id}\t{json.dumps(matrix.tolist())}\n".encode("utf-8"))
    f1.close()

def gen_train_val_data(train_df):
    """
    通过train_test_split()得到训练集和验证集
    :param train_df: 
    :return: 
    """
    y = train_df["Sentiment"]  # <Series>. shape: (156060,)
    y = np_utils.to_categorical(y)  # <ndarray of ndarray>. shape: (156060, 5)
    assert y.shape[1] == 5

    X = train_df["Phrase_vec"]  # <Series>. shape: (156060,)
    X = np.array([json.loads(vec) for vec in X])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.333, shuffle=True, random_state=1)
    # X_train: ndarray of ndarray.  y_train: ndarray.
    return X_train, X_val, y_train, y_val


def gen_train_val_test_data():  # only for vector, not for matrix. IGNORE THIS METHOD.
    """
    :return: X_train, X_val, X_test, X_test_id, y_train, y_val
    """
    train_df = pd.read_csv("../data/output/train_vector_lower.csv", sep="\t")  # (156060, 2)
    # train_df此处不需要去重, 去重的工作在生成word vector之前就完成了

    X_train, X_val, y_train, y_val = gen_train_val_data(train_df)

    test_df = pd.read_csv("../data/output/test_vector_lower.csv", sep="\t")  # (156060, 2)
    X_test = test_df["Phrase_vec"]  # <Series>. shape: (,)
    X_test = np.array([json.loads(vec) for vec in X_test])
    X_test_id = test_df["PhraseId"]  # <Series>. shape: (,)
    # X_test_id = np.array(X_test_id)   # Keep X_test_id in <Series>.
    return X_train, X_val, X_test, X_test_id, y_train, y_val


def gen_train_val_test_matrix():
    """
    :return: X_train, X_val, X_test, X_test_id, y_train, y_val
    """
    start_time = time.time()

    # train_df = pd.read_csv("../data/output/train_matrix_lower_pad.csv", sep="\t")  # (156060, 2)
    train_df = pd.read_csv("../data/output/train_matrix_pad.csv", sep="\t")  # (156060, 2)

    y = train_df["Sentiment"]  # <Series>. shape: (156060,)
    y = np_utils.to_categorical(y)  # <ndarray of ndarray>. shape: (156060, 5)
    assert y.shape[1] == 5

    X = train_df["Phrase_vec"]  # <Series>.
    X = np.array([json.loads(mat) for mat in X])  # shape: (156060, max_phrase_length, vector_size). (150929, 24, 300)

    print(f"X.shape: {X.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.333, shuffle=True, random_state=1)
    # X_train: <ndarray of ndarray>.
    # y_train: <ndarray of ndarray>.

    # test_df = pd.read_csv("../data/output/test_matrix_lower_pad.csv", sep="\t")  # (156060, 2)
    test_df = pd.read_csv("../data/output/test_matrix_pad.csv", sep="\t")  # (156060, 2)
    X_test = test_df["Phrase_vec"]  # <Series>.
    X_test = np.array([json.loads(mat) for mat in X_test])  # shape: (63954, 24, 300)
    X_test_id = test_df["PhraseId"]  # <Series>.
    # X_test_id = np.array(X_test_id)   # Keep X_test_id in <Series>.
    end_time = time.time()
    print(f"Preparing data costs: {end_time - start_time:.2f}s\n")
    return X_train, X_val, X_test, X_test_id, y_train, y_val


def data2vec_bow():
    max_len = 0
    word_freqs = collections.Counter()
    sample_count = 0
    stopwords_set = {"--", "<", ">", ",", ".", "\"", "/", "~", "`", "-", "=", "+", "(", ")", "*", ":", ";", "“", "”",
                     "[", "]", "a", "an", "be", "the", 'of', 'and', 'to', "'s", 'in', 'is', 'that', 'it', 'as', 'with',
                     'for', 'its', 'film', 'movie', 'this', 'you', 'on', 'by', 'his', 'about', 'one', '``', 'at', 'or',
                     'from', 'have', 'are', 'has', "'", 'story', '-rrb-', 'out', 'who', 'into', 'up', '-lrb-', 'if',
                     'what', 'their', '...', 'characters', 'can', 'i', 'your', 'time', 'some', 'does', 'will', 'way',
                     'life', 'been', 'make', 'which', 'he', 'movies', 'do', 'there', 'work', 'her', 'was', 'us', 'own',
                     'they', 'other', 'something', 'would', 'we', 'director', 'through', 'many', 'people', 'when',
                     'made', 'two', 'makes', 'them', 'how', 'action', 'may', 'plot', 'films', 'could', 'character',
                     'see', 'being', 'world', 'audience', 'drama', 'look', 'those', 'sense', 'every', 'another',
                     'should', 'over', "'re", 'feel', 'get', 'minutes', 'man', 'performances', 'cast', 'hollywood',
                     'while', 'human', 'between', 'performance', 'might', 'screen', 'things', 'were', 'had', 'these',
                     'moments', 'script', 'family', 'also', 'seen', 'our', 'before', 'american', 'because', 'watch',
                     'heart', 'end', 'my', 'actors', 'after', 'here', 'cinema', 'go'}

    # 1. 统计句子最长长度、词频统计、单词索引映射
    with open("../data/input/train.tsv", "r") as f:
        f.readline()
        for line in f:
            line_list = line.strip().split("\t")  # split()要求必须是str类型，不能是bytes类型
            sentence = line_list[2]
            words = nltk.word_tokenize(sentence.lower())  # type(words): list
            words = [word for word in words if word not in stopwords_set]  # 去重停用词
            length = len(words)
            if length > max_len:
                max_len = length
            for word in words:
                word_freqs[word] += 1
            sample_count += 1

    print(f"Length of the longest sentence in the training set: {max_len}")  # 53
    print(f"vocabulary size: {len(word_freqs)}")  # 16540. 包括标点符号

    # vocab_size = len(word_freqs)
    # print([item[0] for item in word_freqs.most_common(vocab_size)])
    # print(word_freqs.most_common(vocab_size))
    # word_freqs.most_common(vocab_size): <list of tuple>. [("i", 4705), ",", 4194, ".": 3558, "the": 3221, ...]
    word2index = {word[0]: idx + 2 for idx, word in enumerate(word_freqs.most_common()) if word[1] > 2}
    word2index["PAD"] = 0  # "PAD"没有实际意义
    word2index["UNK"] = 1
    # vocab_size += 2  # 加上"PAD", "UNK"
    vocab_size = len(word2index)
    print(f"vocab_size: {vocab_size}")
    # index2word = {v: k for k, v in word2index.items()}

    # 2. 处理得到训练集和验证集数据
    X, y = bow(sample_count, word2index, max_len)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.333, random_state=1, shuffle=True)

    # 3. 处理得到测试集数据
    test_df = pd.read_csv("../data/input/test.tsv", sep="\t")  # (156060, 2)
    test_sample_count = test_df.shape[0]
    X_test = np.empty(test_sample_count, dtype=list)  # <ndarray of list>
    idx = 0
    for sentence in test_df["Phrase"]:
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X_test[idx] = seqs
        idx += 1

    X_test = sequence.pad_sequences(X_test, maxlen=max_len, value=0)  # default: 从前面补0, 从前面删除 maxlen=MAX_SENTENCE_LENGTH
    X_test_id = test_df["PhraseId"]  # <Series>. shape: (,)
    # X_test_id = np.array(X_test_id)   # Keep X_test_id in <Series>.

    # return X, y, X_test, X_test_id, vocab_size, max_len
    """
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    """
    return X_train, X_val, X_test, X_test_id, y_train, y_val, vocab_size, max_len

def bow(sample_count, word2index, max_len):
    X = np.empty(sample_count, dtype=list)  # <ndarray of list>
    y = np.zeros(sample_count)
    idx = 0
    with open("../data/input/train.tsv", "r") as f:
        f.readline()
        for line in f:
            line_list = line.strip().split("\t")
            label = line_list[3]
            sentence = line_list[2]
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[idx] = seqs
            y[idx] = int(label)
            idx += 1

    X = sequence.pad_sequences(X, maxlen=max_len, value=0)  # default: 从前面补0, 从前面删除
    # 从后面补0, 从后面删除. NOTE: 改成从后面补零和截取后，结果变差了一点.
    # X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, value=0, padding="post", truncating="post")
    y = np_utils.to_categorical(y)
    assert y.shape[1] == 5
    return X, y


if __name__ == "__main__":
    # data2vec_bow()

    """
    origin_train_path = "../data/input/train.tsv"
    origin_test_path = "../data/input/test.tsv"
    train_df, test_df = fetch_data_df(train_path= origin_train_path, test_path=origin_test_path, sep="\t")

    # 1. 去除phrase中的stopwords, 生成文件"../data/output/train_wo_sw.csv" 和 "test_wo_sw.csv"
    rm_stopwords(train_df, test_df)

    # data_analysis(train_df, test_df)
    """

    # train_path = "../data/output/train_wo_sw.csv"  # DEBUG: "train_wo_sw_uniq.csv"
    # test_path = "../data/output/test_wo_sw.csv"
    train_path = "../data/input/train.tsv"
    test_path = "../data/input/test.tsv"
    train_df, test_df = fetch_data_df(train_path=train_path, test_path=test_path, sep="\t")
    train_uniq_flag = False  # True. 只运行一次即可. 以后都设置为False
    if train_uniq_flag:
        print("Before drop_duplicates(), train_df.shape:", train_df.shape)  #  (156060, 2)
        train_df.drop_duplicates(inplace=True)
        print("After drop_duplicates(), train_df.shape:", train_df.shape)  # (106507, 2)
        train_df.to_csv("../data/output/train_wo_sw_uniq.csv", index=False, sep="\t")

    # data2vec(train_df, test_df)
    data2matrix(train_df, test_df)

    # train_df = pd.read_csv("../data/output/train_vector_100.csv", sep="\t")  # (156060, 2)
    # X_train, X_val, y_train, y_val = gen_train_val_data(train_df)

    # gen_train_val_test_data()
    # gen_train_val_test_matrix()
    """
    """

