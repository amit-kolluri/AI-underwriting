import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def convert_text(input_para):
    token_pos_list = []
    tokenized_sents = sent_tokenize(input_para)
    for sent in tokenized_sents:
        tokens = word_tokenize(sent)
        tokens_pos = pos_tag(tokens)
        token_pos_list = token_pos_list + list(tokens_pos)
    return [[s, ] for s in token_pos_list]


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def split_data(X, y, testset_size=0.3):
    """Split train-test data"""
    return train_test_split(X, y, test_size=testset_size, random_state=0)


def train_model(X_train, y_train):
    # Create CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    # Fit CRF model
    crf.fit(X_train, y_train)

    return crf


def evaluate_performance(y_test, y_pred, avg='weighted'):
    """evaluate model performance"""
    accuracy = metrics.flat_f1_score(y_test, y_pred, average=avg)
    precison = metrics.flat_precision_score(y_test, y_pred, average=avg)
    recall = metrics.flat_recall_score(y_test, y_pred, average=avg)
    f1score = metrics.flat_f1_score(y_test, y_pred, average=avg)
    return {"accuracy": accuracy, "precision": precison, "recall": recall, "f1score": f1score}


def load_dataset(filepath):
    """ Load dataset """
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    return df


def make_prediction(model, X_test):
    results = []
    y_pred = model.predict(X_test)
    for i, y_tag in enumerate(y_pred):
        if y_tag[0] != 'O':
            results.append({'text': X_test[i][0]['word.lower()'], 'entity': y_tag[0]})
    return results
