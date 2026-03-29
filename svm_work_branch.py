import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#Written with the help of AI

df = pd.read_csv('qprop_data/proppy_1.0.train.tsv', sep='\t', header=None)
df.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['propaganda_label'] = np.where(df['propaganda_label'] == -1, 0, 1)

# Clean
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

df['clean_text'] = df['article_text'].apply(advanced_clean)

# Vectorize
vectorizer = CountVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(df['clean_text'])
y_train = df['propaganda_label']
vocab = vectorizer.get_feature_names_out()

# Training
print("Training")

# Initialize the model
svm_model = LinearSVC(random_state=42, max_iter=2000)

svm_model.fit(X_train, y_train)

#results
print("Weight Results")

svm_weights = svm_model.coef_[0]
word_weights = list(zip(vocab, svm_weights))
word_weights.sort(key=lambda x: x[1])

print("\nTop 10 Strongest 'Real News' Indicators:")
for word, weight in word_weights[:10]:
    print(f"   {word}: {weight:.4f}")

print("\nTop 10 Strongest 'Propaganda' Indicators:")
for word, weight in reversed(word_weights[-10:]):
    print(f"   {word}: {weight:.4f}")