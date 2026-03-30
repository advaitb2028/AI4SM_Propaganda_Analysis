import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. THE NUCLEAR RESET ---
# This command forces VS Code to close every single open image window
plt.close('all')

# --- 2. LOAD DATA ---
print("Step 1: Loading Data...")
try:
    df = pd.read_csv("3271522/proppy_1.0.train.tsv", sep='\t', header=None)
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
vectorizer = TfidfVectorizer(
    max_features = 500,
    stop_words='english',
    ngram_range=(1, 2),

)

#max_df=0.97

X = vectorizer.fit_transform(df['clean_text'])


y_binary = np.where(df['propaganda_label'] == -1, 0, 1)

# Convert the sparse matrix X into a normal numpy array
X_dense = X.toarray()

# 2. Manual Train/Test Split (80% Train, 20% Test)
split_idx = int(0.8 * len(X_dense))
X_train, X_test = X_dense[:split_idx], X_dense[split_idx:]
y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]


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

predict_knn(X_train, y_train, X_test)

y_test = np.array(y_test)

pred_1 = np.array(pred_1)
pred_3 = np.array(pred_3)
pred_5 = np.array(pred_5)
pred_10 = np.array(pred_10)



print(pred_1[0].dtype)
print(y_test[0].dtype)


accuracy = np.mean(pred_1 == y_test)
true_pos = np.sum((pred_1 == 1) & (y_test == 1))
false_pos = np.sum((pred_1 == 1) & (y_test == 0))
false_neg = np.sum((pred_1 == 0) & (y_test == 1))
precision_1 = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall_1 = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
print(f"K=1: Precision: {precision_1}, Recall: {recall_1}, F1: {f1_1} ,Accuracy: {accuracy * 100}%")

accuracy = np.mean(pred_3 == y_test)
true_pos = np.sum((pred_3 == 1) & (y_test == 1))
false_pos = np.sum((pred_3 == 1) & (y_test == 0))
false_neg = np.sum((pred_3 == 0) & (y_test == 1))
precision_3 = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall_3 = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1_3 = 2 * (precision_3 * recall_3) / (precision_3 + recall_3) if (precision_3 + recall_3) > 0 else 0
print(f"K=3: Precision: {precision_3}, Recall: {recall_3}, F1: {f1_3} ,Accuracy: {accuracy * 100}%")

accuracy = np.mean(pred_5 == y_test)
true_pos = np.sum((pred_5 == 1) & (y_test == 1))
false_pos = np.sum((pred_5 == 1) & (y_test == 0))
false_neg = np.sum((pred_5 == 0) & (y_test == 1))
precision_5 = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall_5 = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1_5 = 2 * (precision_5 * recall_5) / (precision_5 + recall_5) if (precision_5 + recall_1) > 0 else 0
print(f"K=5: Precision: {precision_5}, Recall: {recall_5}, F1: {f1_5} ,Accuracy: {accuracy * 100}%")

accuracy = np.mean(pred_10 == y_test)
true_pos = np.sum((pred_10 == 1) & (y_test == 1))
false_pos = np.sum((pred_10 == 1) & (y_test == 0))
false_neg = np.sum((pred_10 == 0) & (y_test == 1))
precision_10 = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall_10 = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1_10 = 2 * (precision_1 * recall_10) / (precision_10 + recall_10) if (precision_10 + recall_10) > 0 else 0
print(f"K=10: Precision: {precision_10}, Recall: {recall_10}, F1: {f1_10} ,Accuracy: {accuracy * 100}%")



import matplotlib.pyplot as plt

# Organizing data for the graph
ks = ['1', '3', '5', '10']
# Note: You'll need to define precision_3, recall_3, etc., using the pattern above
accuracies = [np.mean(pred_1 == y_test), np.mean(pred_3 == y_test), np.mean(pred_5 == y_test), np.mean(pred_10 == y_test)]
precisions = [precision_1, precision_3, precision_5, precision_10]
recalls = [recall_1, recall_3, recall_5, recall_10]
f1s = [f1_1, f1_3, f1_5, f1_10]

x = np.arange(len(ks))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*width, accuracies, width, label='Accuracy')
ax.bar(x - 0.5*width, precisions, width, label='Precision')
ax.bar(x + 0.5*width, recalls, width, label='Recall')
ax.bar(x + 1.5*width, f1s, width, label='F1')

ax.set_xlabel('Value of K')
ax.set_ylabel('Score')
ax.set_title('KNN Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(ks)
ax.legend()

plt.show()

print("Done.")
