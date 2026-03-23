import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATA ---
print("Step 1: Loading Data...")
df = pd.read_csv('qprop_data/proppy_1.0.train.tsv', sep='\t', header=None)
df.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# The math needs 0 (News) and 1 (Propaganda). Convert -1s to 0s.
df['propaganda_label'] = np.where(df['propaganda_label'] == -1, 0, 1)

# --- 2. ADVANCED CLEANING (Stemming & Regex) ---
print("Step 2: Cleaning Text & Stemming...")
nltk.download('stopwords', quiet=True)
stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = PorterStemmer()

def advanced_clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

df['clean_text'] = df['article_text'].apply(advanced_clean)

# --- 3. VECTORIZING ---
print("Step 3: Building Vectors...")
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text'])
vocab = vectorizer.get_feature_names_out()

# --- 4. TRAIN/TEST SPLIT ---
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = df['propaganda_label'][:split_idx], df['propaganda_label'][split_idx:]

# --- 5. THE SVM MODEL ---
print("\n" + "="*40)
print("Step 5: Training Support Vector Machine (SVM)")
print("="*40)

# Initialize the model
svm_model = LinearSVC(random_state=42, max_iter=2000)

# Train the model (This replaces your gradient descent loop!)
print("Training model... (This is lightning fast!)")
svm_model.fit(X_train, y_train)

# Grade the model
test_predictions = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
print(f"\n SVM Accuracy: {accuracy * 100:.2f}%")

# --- 6. PEEKING INTO THE SVM'S BRAIN ---
print("\n" + "="*40)
print("Step 6: Interpreting the SVM Weights")
print("="*40)

# Extract the weights (coefficients) from the SVM
svm_weights = svm_model.coef_[0]
word_weights = list(zip(vocab, svm_weights))
word_weights.sort(key=lambda x: x[1])

print("\nTop 10 Strongest 'Real News' Indicators (-):")
for word, weight in word_weights[:10]:
    print(f"   {word}: {weight:.4f}")

print("\nTop 10 Strongest 'Propaganda' Indicators (+):")
for word, weight in reversed(word_weights[-10:]):
    print(f"   {word}: {weight:.4f}")