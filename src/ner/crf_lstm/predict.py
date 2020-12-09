from keras.models import load_model
from .train import create_model
from keras.preprocessing.sequence import pad_sequences
# from keras_contrib.utils import save_load_utils # not using could be removed
import numpy as np
from keras_contrib.layers import CRF
import pickle


tags = ['B-nat', 'I-eve', 'I-art', 'B-art', 'I-tim', 'I-nat', 'O', 'B-tim', 'B-gpe', 'I-gpe', 'B-eve', 'B-per',
        'I-org', 'B-org', 'B-geo', 'I-geo', 'I-per']
with open("src/ner/crf_lstm/words.pickle", "rb") as words:
    word2idx = pickle.load(words)

path = "src/ner/models/"

max_len = 75
n_words = len(word2idx)
n_tags = len(tags)


def create_custom_objects():
    instanceHolder = {"instance": None}

    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)

    return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "crf_loss": loss, "crf_viterbi_accuracy": accuracy}


def crf_lstm_prediction(test_sentence, model_path=path):
    predictions = []
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence.split()]],
                                padding="post", value=0, maxlen=max_len)
    model = create_model(n_tags, max_len, n_words)
    # model_loaded = save_load_utils.load_all_weights(model, path)
    model_loaded = load_model(path + "crf_lstm_model", custom_objects=create_custom_objects())
    p = model_loaded.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    # print("{:15}||{}".format("Word", "Prediction"))
    # print(30 * "=")
    for w, pred in zip(test_sentence.split(), p[0]):
        predict = {'entity': tags[pred], 'text': w,
                   'start_position': 0, 'confidence': 1,
                   'end_position': 0}
        predictions.append(predict)
        # print("{:15}: {:5}".format(w, tags[pred]))
        # print("word:", w, tags[pred])
    return predictions


if __name__ == "__main__":
    test_sentence = """
    Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
    reporter for the network, about protests in Minnesota and elsewhere. 
    """
    results = crf_lstm_prediction(test_sentence, path)
    print(results)
