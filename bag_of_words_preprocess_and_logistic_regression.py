import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

#Written with the help of AI

df_train = pd.read_csv('qprop_data/proppy_1.0.train.tsv', sep='\t', header=None)
df_train.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

df_train['propaganda_label'] = np.where(df_train['propaganda_label'] == -1, 0, 1)

# clean
nltk.download('stopwords', quiet=True)
stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = PorterStemmer()

def advanced_clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

df_train['clean_text'] = df_train['article_text'].apply(advanced_clean)

# vectorize
vectorizer = CountVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(df_train['clean_text'])
vocab = vectorizer.get_feature_names_out()

# model

# Convert the sparse matrix into a normal numpy array for the custom math
X_train_dense = X_train.toarray()
y_train = df_train['propaganda_label'].values

print(f"Training on {len(X_train_dense)} articles...")

# math functions
def sigmoid(z):
    z = np.clip(z, -500, 500) 
    return 1.0 / (1.0 + np.exp(-z))

def custom_logistic_regression(X, y, learning_rate=0.5, epochs=100):
    m, n = X.shape 
    weights = np.zeros(n)
    bias = 0.0
    
    print(f"running {epochs} times")
    for epoch in range(epochs):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
            
    return weights, bias

# predict
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return [1 if p > 0.5 else 0 for p in predictions]

# training
final_weights, final_bias = custom_logistic_regression(X_train_dense, y_train, learning_rate=0.5, epochs=100)

# Results
print("Weight Results:")

word_weights = list(zip(vocab, final_weights))
word_weights.sort(key=lambda x: x[1])

print("Top 10 Strongest 'Real News' Indicators:")
for word, weight in word_weights[:10]:
    print(f"   {word}: {weight:.4f}")

print("\nTop 10 Strongest 'Propaganda' Indicators:")
for word, weight in reversed(word_weights[-10:]):
    print(f"   {word}: {weight:.4f}")

#RUN ON TEST DATASET

df_test = pd.read_csv('qprop_data/proppy_1.0.test.tsv', sep='\t', header=None)
df_test.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
df_test['propaganda_label'] = np.where(df_test['propaganda_label'] == -1, 0, 1)
df_test['clean_text'] = df_test['article_text'].apply(advanced_clean)

X_test = vectorizer.transform(df_test['clean_text'])

X_test_dense = X_test.toarray()
y_test = df_test['propaganda_label'].values

log_predictions = predict(X_test_dense, final_weights, final_bias)

print("Logistic Regression Results")
print(f"Accuracy:    {accuracy_score(y_test, log_predictions) * 100:.2f}%")
print(f"Precision:   {precision_score(y_test, log_predictions) * 100:.2f}%")
print(f"Recall:      {recall_score(y_test, log_predictions) * 100:.2f}%")
print(f"F1 (Macro):  {f1_score(y_test, log_predictions, average='macro') * 100:.2f}%")
print(f"F1 (Weight): {f1_score(y_test, log_predictions, average='weighted') * 100:.2f}%")    