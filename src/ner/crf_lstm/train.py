import pandas as pd
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
# from keras_contrib.utils import save_load_utils
# from tf2crf import CRF
from keras_contrib.layers import CRF
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

max_len = 75


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        def agg_func(s): return [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
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


def prepare_data(word2idx, tag2idx, n_tags, sentences, max_len):
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y,
                      padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
    return X_tr, X_te, y_tr, y_te


def create_model(n_tags, max_len, n_words):
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(
        model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function,
                  metrics=[crf.accuracy])
    return model


def train_model(model, X_tr, y_tr):
    history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5,
                        validation_split=0.1, verbose=1)
    return history


def save_model(model):
    file_name = "C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\" \
                "models\\crf_lstm_model"
    # save the model
    model.save(file_name)
    # save_load_utils.save_all_weights(model, file_name)


def pred2label(pred, idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
            out.append(out_i)
    return out


if __name__ == "__main__":
    data = pd.read_csv("C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src"
                       "\\ner\\data\\ner_dataset.csv", encoding="latin1")
    data = data.fillna(method="ffill")
    getter = SentenceGetter(data)
    sentences = getter.sentences

    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    n_words = len(words)

    tags = list(set(data["Tag"].values))
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    with open("words.pickle", "wb") as word:
        pickle.dump(word2idx, word)
    print(word2idx)
    tag2idx = {t: i for i, t in enumerate(tags)}
    print(tags)
    n_tags = len(tags)
    X_tr, X_te, y_tr, y_te = prepare_data(word2idx, tag2idx, n_tags, sentences, max_len)
    model = create_model(n_tags, max_len, n_words)
    model.summary()
    history = train_model(model, X_tr, y_tr)
    test_pred = model.predict(X_te, verbose=1)
    idx2tag = {i: w for w, i in tag2idx.items()}
    #
    # pred_labels = pred2label(test_pred)
    # test_labels = pred2label(y_te)

    # print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    # print(classification_report(test_labels, pred_labels))
