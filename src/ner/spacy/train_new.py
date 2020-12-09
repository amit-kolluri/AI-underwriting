import spacy
import random
from pathlib import Path
import pickle

output_dir = Path("D:\\Travelers\\data\\model1")
n_iter = 100
model = input('Enter model name:')
TRAIN_DATA = None
# TRAIN_DATA = [
#     ('Name of the Applicant: XYZ', {
#         'entities': [(23, 25, 'NAME')]
#     }),
#      ('Name of the Applicant: Divakar Sharma', {
#         'entities': [(23, 36, 'NAME')]
#     }),
#     ('Name of the Applicant: Health research pvt lt.', {
#         'entities': [(23, 46, 'NAME')]
#     })
# ]


def train_spacy(data, iterations):
    TRAIN_DATA = data
    if model:
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


if TRAIN_DATA:
    prdnlp = train_spacy(TRAIN_DATA, n_iter)
else:
    with open('D:\\Travelers\\data\\ner_corpus_260', 'rb') as fp:
        train_data = pickle.load(fp)
        prdnlp = train_spacy(train_data, n_iter)

# Save our trained Model
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    prdnlp.to_disk(output_dir)
    print("Saved model to", output_dir)
