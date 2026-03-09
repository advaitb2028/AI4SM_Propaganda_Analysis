import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. THE NUCLEAR RESET ---
# This command forces VS Code to close every single open image window
plt.close('all')

# --- 2. LOAD DATA ---
print("Step 1: Loading Data...")
try:
    df = pd.read_csv("/Users/advaitbhowmik/downloads/3271522/proppy_1.0.train.tsv", sep='\t', header=None)
    df.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
except FileNotFoundError:
    print("File not found.")
    exit()

# --- 3. CLEANING ---
print("Step 2: Cleaning Text...")
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def simple_clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return " ".join([w for w in text.split() if w not in stop_words])

df['clean_text'] = df['article_text'].apply(simple_clean)

# --- 4. VECTORIZING ---
print("Step 3: Building Vectors...")
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text'])
vocab = vectorizer.get_feature_names_out()

print(df.columns)

y_binary = np.where(df['propaganda_label'] == -1, 0, 1)

# Convert the sparse matrix X into a normal numpy array
X_dense = X.toarray()

# 2. Manual Train/Test Split (80% Train, 20% Test)
split_idx = int(0.8 * len(X_dense))
X_train, X_test = X_dense[:split_idx], X_dense[split_idx:]
y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]

print(X_train[0])

## WRONG

import heapq

def calculate_distances(test_row):
    distances = []
    heapq.heapify(distances)
    for i in range(len(X_train)):
        distance = sum((X_train[i] - test_row) ** 2) ** 0.5
        heapq.heappush(distances, (distance, i))
    return distances

pred_y = []
def get_neighbors(neighbor_distances):
    k = 1
    votes = {}
    for i in range(k):
      _, index = heapq.heappop(distances)
      value = y_train[index]
      if value in votes:
        votes[value] += 1
      else:
        votes[value] = 1

    test_y = None
    most_votes_label = max(votes.values())
    for key in votes:
      if votes[key] == most_votes_label:
        test_y = key
        #print(f"LABEL: {test_y}")
        pred_y.append(test_y)
        break


for i in range(1000):
    test_row = X_test[i]
    distances = calculate_distances(test_row)
    get_neighbors(distances)

correct = 0.0
print(len(pred_y))
for i in range(len(pred_y)):
    if pred_y[i] == y_test[i]:
        correct += 1

print(correct)
print(correct / 20)

## ^^^^^^^  WRONG

import numpy as np

pred_1 = []
pred_3 = []
pred_5 = []
pred_10 = []


def predict_knn(X_train, y_train, X_test):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    predictions = []

    for test_row in X_test:
        distances = np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))

        neighbor_indices = np.argpartition(distances, 1)[:1]
        neighbor_labels = y_train[neighbor_indices]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        pred_1.append(labels[np.argmax(counts)])

        neighbor_indices = np.argpartition(distances, 3)[:3]
        neighbor_labels = y_train[neighbor_indices]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        pred_3.append(labels[np.argmax(counts)])

        neighbor_indices = np.argpartition(distances, 5)[:5]
        neighbor_labels = y_train[neighbor_indices]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        pred_5.append(labels[np.argmax(counts)])

        neighbor_indices = np.argpartition(distances, 10)[:10]
        neighbor_labels = y_train[neighbor_indices]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        pred_10.append(labels[np.argmax(counts)])


print("Starting Compute")

predict_knn(X_train, y_train, X_test[:100])

accuracy = np.mean(pred_1 == y_test[:100])
print(f"Accuracy: {accuracy * 100}%")

accuracy = np.mean(pred_3 == y_test[:100])
print(f"Accuracy: {accuracy * 100}%")

accuracy = np.mean(pred_5 == y_test[:100])
print(f"Accuracy: {accuracy * 100}%")

accuracy = np.mean(pred_10 == y_test[:100])
print(f"Accuracy: {accuracy * 100}%")

print("Done.")