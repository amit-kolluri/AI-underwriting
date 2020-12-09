import numpy as np
import pandas as pd
import torch
from transformers import BertForTokenClassification, BertTokenizer

path = 'src/ner/models'


def bert_prediction(text, model_path=path):
    tag_values = ['B-nat', 'I-eve', 'I-art', 'B-art', 'I-tim', 'I-nat', 'O', 'B-tim', 'B-gpe', 'I-gpe', 'B-eve',
                  'B-per', 'I-org', 'B-org', 'B-geo', 'I-geo', 'I-per', "PAD"]
    prediction = []
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=18,
        output_attentions=False,
        output_hidden_states=False)
    model.load_state_dict(torch.load(
        model_path + "model_bert.pt", map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=False)
    tokenized_sentence = tokenizer.encode(text)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(
            input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)

        for token, label in zip(new_tokens, new_labels):
            predict = {'entity': label, 'text': token,
                       'start_position': 0, 'confidence': 1,
                       'end_position': 0}
            prediction.append(predict)
            # print("{}\t{}".format(label, token))

    return prediction


if __name__ == '__main__':
    test_sentence = """
    Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a 
    reporter for the network, about protests in Minnesota and elsewhere. 
    """
    results = bert_prediction(test_sentence, path)
    print(results)
