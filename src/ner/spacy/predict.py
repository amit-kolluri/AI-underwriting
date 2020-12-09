import spacy

path = 'C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\models\\'


def spacy_prediction(text, model_path=path, model_name="spacy_blank"):
    prediction = []
    if "spacy" in model_path:
        nlp = spacy.load(model_path)
    else:
        nlp = spacy.load(model_path + model_name)
    result = nlp(text)
    for value in result.ents:
        predict = {'entity': value.label_, 'text': value.text,
                   'start_position': value.start_char, 'confidence': 1,
                   'end_position': value.end_char}
        prediction.append(predict)
    return prediction


if __name__ == '__main__':
    test_sentence = """
    Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
    reporter for the network, about protests in Minnesota and elsewhere. 
    """
    results = spacy_prediction(test_sentence, path)
    print(results)
