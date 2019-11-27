import tensorflow as tf
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

stemmer = SnowballStemmer("russian")

def prepare_data_as_csv():
    f_train_enhanced = open("train_sentences_enhanced.txt", "r")
    train_sentences_enhanced = f_train_enhanced.read().split("\n")
    f_train = open("train_sentences.txt", "r")
    train_sentences = f_train.read().split("\n")
    f_train_tags = open("train_nes.txt", "r")
    train_tags = f_train_tags.read().split("\n")

    train_data = []
    num_sent = 1

    for sentence in train_sentences_enhanced:
        # if num_sent == 8:
        #    break

        tokens = word_tokenize(sentence)
        n = len(tokens)
        i = 0

        while i < n:
            elem = []
            elem.append("Sentence: " + str(num_sent))
            elem.append(stemmer.stem(tokens[i]))
            elem.append("POS")

            if i + 3 < n and tokens[i + 1] == "{" and tokens[i + 3] == "}":
                if tokens[i + 2] == "ORG":
                    elem.append("ORG")
                    i += 4
                elif tokens[i + 2] == "PERSON":
                    elem.append("PERSON")
                    i += 4
                else:
                    elem.append("O")
                    i += 1
            else:
                elem.append("O")
                i += 1

            train_data.append(elem)

        num_sent += 1

    df = pd.DataFrame(train_data, columns=['Sentence #', 'Word', 'POS', 'Tag'])
    return df


def get_test_dataset():
    f_test = open("dataset_40163_1.txt", "r")
    test_sentences = f_test.read().split("\n")

    test_data = []
    test_data_unlemm = []

    for sentence in test_sentences:
        tokens = word_tokenize(sentence)
        n = len(tokens)
        i = 0

        while i < n:
            elem = []
            elem.append("Sentence: " + str(num_sent))
            elem.append(stemmer.stem(tokens[i]))
            elem.append("POS")
            elem.append("O")
            test_data.append(elem)

            elem = []
            elem.append("Sentence: " + str(num_sent))
            elem.append(tokens[i])
            elem.append("POS")
            elem.append("O")
            test_data_unlemm.append(elem)

        num_sent += 1

    df = pd.DataFrame(test_data, columns=['Sentence #', 'Word', 'POS', 'Tag'])
    t_getter = SentenceGetter(df)
    t_sentences = t_getter.sentences
    X_te = [[word2idx[w[0]] for w in s] for s in t_sentences]

    df = pd.DataFrame(test_data_unlemm, columns=['Sentence #', 'Word', 'POS', 'Tag'])
    t_getter = SentenceGetter(df)
    unlemmatized_X_te = t_getter.sentences

    return X_te, unlemmatized_X_te

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


if __name__ == "__main__":
    if tf.test.is_gpu_available():
        BATCH_SIZE = 512
        EPOCHS = 5
        MAX_LEN = 85
        EMBEDDING = 40
    else:
        BATCH_SIZE = 32
        EPOCHS = 5
        MAX_LEN = 85
        EMBEDDING = 20

    data = prepare_data_as_csv()
    data = data.fillna(method="ffill")
    words = list(set(data["Word"].values))
    n_words = len(words)
    tags = list(set(data["Tag"].values))
    n_tags = len(tags)

    getter = SentenceGetter(data)
    sentences = getter.sentences

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    idx2word = {i: w for w, i in word2idx.items()}
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    idx2tag = {i: w for w, i in tag2idx.items()}

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])
    y = [to_categorical(i, num_classes=n_tags + 1) for i in y]
    X_tr, y_tr = X, y

    input = Input(shape=(MAX_LEN,))
    model = Embedding(input_dim=n_words + 2, output_dim=EMBEDDING,
                      input_length=MAX_LEN, mask_zero=True)(input)
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation="relu"))(model)
    crf = CRF(n_tags + 1)
    out = crf(model)
    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_split=0.1, verbose=2)

    X_te, unlemmatized_X_te = get_test_dataset()
    ans = []
    cur_ind = 0

    for i in range(len(X_te)):
        p = model.predict(np.array([X_te[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(y_te[i], -1)

        cur_ans = []

        for w, unlem_w, t, pred in zip(X_te[i], unlemmatized_X_te[i], true, p[0]):
            if w != 0:
                if idx2tag[pred] == "ORG" or idx2tag[pred] == "PERSON":
                    cur_ans.add(str(cur_ind) + " " + str(len(unlem_w)) + " " + idx2tag[pred])

            cur_ind += str(len(unlem_w)) + 1

        cur_ans.append("EOL")
        # print(cur_ans)
        ans.append(" ".join(cur_ans))

    print(len(ans))
    f_ans = open("ans.txt", "w")
    f_ans.write("\n".join(ans))
    f_ans.close()
