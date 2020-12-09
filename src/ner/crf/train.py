from helper import SentenceGetter
from helper import convert_text
from helper import word2features
from helper import sent2features
from helper import sent2labels
from helper import sent2tokens
from helper import split_data
from helper import train_model
from helper import evaluate_performance
from helper import load_dataset
from helper import make_prediction
import joblib

# Load dataset
filepath = 'src/ner/data/ner_dataset.csv'
df = load_dataset(filepath)

# Get sentences
getter = SentenceGetter(df)
sentences = getter.sentences

# Get features and labels
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
print("training data", X, y)
# Train model
print("Training model")
# crf = train_model(X, y)
#
# # Save the model as a pickle in a file
# joblib.dump(crf, 'model/crf_model.pkl')

# Predict the labels
# y_pred = crf.predict(X)

# Evaluate the performance
# results = evaluate_performance(y, y_pred)

# print("Accuracy on train dataset", results)

# ########################################################====================================================####################

# # Split data
# X_train, X_test, y_train, y_test = split_data(X,y)

# # Train model
# crf=train_model(X, y)

# # Predict the labels
# y_pred = crf.predict(X_test)

# # Evaluate the performance
# results=evaluate_performance(y_test, y_pred)

# print("Accuracy on test dataset",results)
