from .helper import convert_text
from .helper import make_prediction
from .helper import sent2features
import joblib


def crf_prediction(text, model_path, model_name):
    # Load the model from the file
    model_name = "crf_model.pkl"
    crf_model = joblib.load(model_path + model_name)
    transformed_text = convert_text(text)
    X_new = [sent2features(s) for s in transformed_text]
    return make_prediction(crf_model, X_new)


if __name__ == '__main__':
    text = text = 'Sukanya, Rajib and Naba are my good friends. Sukanya is getting married next year. Marriage is a ' \
                  'big step in one’s life. \nIt is both exciting and frightening. But friendship is a sacred bond between ' \
                  'people. It is a special kind of love between us.  \nMany of you must have tried searching for a ' \
                  'friend but never found the right one.'

    results = crf_prediction(text)
    print(results)
