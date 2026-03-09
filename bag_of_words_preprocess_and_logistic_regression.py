import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem import PorterStemmer
import random

# --- 1. THE NUCLEAR RESET ---
# This command forces VS Code to close every single open image window
plt.close('all') 

# --- 2. LOAD DATA ---
print("Step 1: Loading Data...")
try:
    df = pd.read_csv('qprop_data/proppy_1.0.train.tsv', sep='\t', header=None)
    df.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
except FileNotFoundError:
    print("File not found.")
    exit()

# --- 3. CLEANING ---
print("Step 2: Cleaning Text (Advanced)...")
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize the Stemmer
stemmer = PorterStemmer()

def advanced_clean(text):
    if not isinstance(text, str): return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove all numbers/digits using regex
    text = re.sub(r'\d+', '', text)
    
    # 3. Remove punctuation
    text = "".join([c for c in text if c not in string.punctuation])
    
    # 4. Tokenize (split), remove stopwords, AND apply stemming in one pass
    words = text.split()
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    # 5. Join back together with a single space
    return " ".join(cleaned_words)

# Apply the new function to your dataframe
df['clean_text'] = df['article_text'].apply(advanced_clean)

# --- 4. VECTORIZING ---
print("Step 3: Building Vectors...")
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text'])
vocab = vectorizer.get_feature_names_out()

print("-"*40)
print("SANITY CHECK: The New Vocabulary")
print("-"*40)

# Convert the vocab array to a list and grab 20 random words
random_vocab_sample = random.sample(list(vocab), 20)

print(f"Total words in vocabulary: {len(vocab)}")
print("Random sample of 20 stemmed words:")
print(random_vocab_sample)
print("-"*40 + "\n")

# --- STEP 6: CUSTOM LOGISTIC REGRESSION FROM SCRATCH ---
print("\n" + "="*40)
print("Step 6: Training Custom Math Model...")
print("="*40)

# 1. Prepare the Data for pure Math
# The math needs 0 (News) and 1 (Propaganda). We must convert the -1s to 0s.
y_binary = np.where(df['propaganda_label'] == -1, 0, 1)

# Convert the sparse matrix X into a normal numpy array
X_dense = X.toarray()

# 2. Manual Train/Test Split (80% Train, 20% Test)
split_idx = int(0.8 * len(X_dense))
X_train, X_test = X_dense[:split_idx], X_dense[split_idx:]
y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]

print(f"Training on {len(X_train)} articles...")

# 3. The Math Functions
def sigmoid(z):
    # np.clip prevents the math from exploding if the numbers get too big
    z = np.clip(z, -500, 500) 
    return 1.0 / (1.0 + np.exp(-z))

def custom_logistic_regression(X, y, learning_rate=0.5, epochs=100):
    m, n = X.shape # m = number of articles, n = vocabulary size (2000)
    
    # Start with all weights and bias set to zero
    weights = np.zeros(n)
    bias = 0.0
    
    print(f"Model is doing pushups ({epochs} rounds)...")
    for epoch in range(epochs):
        # A. Make a Guess (Forward Pass)
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        
        # B. Calculate the mistakes (Gradient Descent)
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        # C. Update the weights to be smarter next time
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"   Round {epoch + 1} complete...")
            
    return weights, bias

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    # If the probability is > 50%, guess 1 (Propaganda), else guess 0 (News)
    return [1 if p > 0.5 else 0 for p in predictions]

# 4. Train the Model!
final_weights, final_bias = custom_logistic_regression(X_train, y_train, learning_rate=0.5, epochs=100)
print("Training Complete!")

# 5. Grade the Model
print("\nGrading the custom model on the test set...")
test_predictions = predict(X_test, final_weights, final_bias)

# Calculate Accuracy manually
correct_guesses = sum(1 for true, pred in zip(y_test, test_predictions) if true == pred)
accuracy = correct_guesses / len(y_test)

print(f"\nLogistic Regression Accuracy: {accuracy * 100:.2f}%")

# --- STEP 7: PEEKING INSIDE THE AI'S BRAIN ---
print("\n" + "="*40)
print("Step 7: Interpreting the Learned Weights")
print("="*40)

# 1. Zip the vocabulary words together with their final learned weights
word_weights = list(zip(vocab, final_weights))

# 2. Sort the list from smallest (most negative) to largest (most positive)
word_weights.sort(key=lambda x: x[1])

# 3. Print the strongest indicators of Real News (Label 0 / Negative weights)
print("\nTop 10 Strongest 'Real News' Indicators:")
for word, weight in word_weights[:10]: # The first 10 items
    print(f"   {word}: {weight:.4f}")

# 4. Print the strongest indicators of Propaganda (Label 1 / Positive weights)
print("\nTop 10 Strongest 'Propaganda' Indicators:")
for word, weight in reversed(word_weights[-10:]): # The last 10 items, reversed
    print(f"   {word}: {weight:.4f}")