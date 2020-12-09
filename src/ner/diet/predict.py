from rasa.nlu.model import Interpreter
from rasa.model import get_model_subdirectories, get_model

path = "C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\models\\"


def diet_prediction(text: str, model_path: str = path):
    prediction = []

    model_path = get_model(model_path + "nlu-20201202-134455.tar.gz")
    _, nlu_model = get_model_subdirectories(model_path)
    print("model_name:", nlu_model)
    interpreter = Interpreter.load(nlu_model)
    print("NLU model loaded.")
    result = interpreter.parse(text)
    for entity in result["entities"]:
        predict = {'entity': entity["entity"], 'text': entity["value"],
                   'start_position': entity["start"], 'confidence': entity["confidence_entity"],
                   'end_position': entity["end"]}
        prediction.append(predict)

    return prediction


if __name__ == "__main__":
    test_sentence = """Trump tweets began just moments after a Fox News report by Mike Tobin, a reporter for the 
     network, about protests in Minnesota and elsewhere."""
    results = diet_prediction(test_sentence, path)
    print(results)
