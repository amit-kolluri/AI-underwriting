import pandas as pd

df = pd.read_csv(
    "C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\data\\ner_dataset.csv", encoding="latin1")
df.drop(['Sentence #', 'POS'], axis=1, inplace=True)
# df = df[:50000]

df.to_csv('C:\\Users\\Divakar.Sharma\\Desktop\\Travelers\\AI-Underwriting\\src\\ner\\data\\ner_corpus_260.tsv', sep='\t', index=None)
