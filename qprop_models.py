"""
QProp Propaganda Detection
Decision Tree + Random Forest — Built Entirely From Scratch
"""

import numpy as np
import pandas as pd
import re
import string
import math
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_data(filepath='proppy_1.0.train.tsv'):
    columns = [
        'article_text', 'event_location', 'average_tone', 'article_date',
        'article_ID', 'article_URL', 'MBFC_factuality_label', 'article_URL_dup',
        'MBFC_factuality_label_dup', 'URL_to_MBFC_page', 'source_name',
        'MBFC_notes_about_source', 'MBFC_bias_label', 'source_URL', 'propaganda_label'
    ]
    df = pd.read_csv(filepath, sep='\t', names=columns, header=None,
                     low_memory=False, encoding='utf-8')
    df['propaganda_label'] = df['propaganda_label'].astype(int)
    if -1 in df['propaganda_label'].unique():
        df['propaganda_label'] = df['propaganda_label'].replace(-1, 0)
    df['article_text'] = df['article_text'].astype(str).str.strip()
    df['average_tone'] = pd.to_numeric(df['average_tone'], errors='coerce').fillna(0)

    print(f"Loaded {len(df):,} articles")
    print(f"  Propaganda:     {df['propaganda_label'].sum():,}")
    print(f"  Non-propaganda: {(df['propaganda_label']==0).sum():,}")

    prop     = df[df['propaganda_label'] == 1]
    non_prop = df[df['propaganda_label'] == 0].sample(n=len(prop), random_state=42)
    df = pd.concat([prop, non_prop]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nBalanced to {len(df):,} articles (50/50)")
    return df


def train_test_split(X, y, test_size=0.2, random_seed=42):
    rng = np.random.RandomState(random_seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def extract_basic_stats(df):
    texts = df['article_text']
    features = np.column_stack([
        texts.apply(lambda x: len(x.split())).values,
        texts.apply(len).values,
        texts.apply(lambda x: len(re.findall(r'[.!?]+', x))).values,
        texts.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0).values,
        texts.apply(lambda x: x.count('!')).values,
        texts.apply(lambda x: x.count('?')).values,
        texts.apply(lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)).values,
        texts.apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0).values,
        texts.apply(lambda x: sum(1 for c in x if c in string.punctuation)).values,
        texts.apply(lambda x: len(re.findall(r'http[s]?://', x))).values,
        texts.apply(lambda x: len(re.findall(r'#\w+', x))).values,
        df['average_tone'].values,
    ])
    names = ['word_count','char_count','sentence_count','avg_word_length',
             'exclamation_count','question_count','all_caps_words',
             'uppercase_ratio','punctuation_count','url_count','hashtag_count','average_tone']
    return features.astype(float), names


def tokenize(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    stopwords = {
        'the','a','an','and','or','but','in','on','at','to','for','of','is',
        'it','its','this','that','was','are','be','been','has','have','had',
        'with','as','by','from','up','about','into','through','during','i',
        'we','you','he','she','they','our','their','his','her','not','no',
        'so','if','than','then','when','there','which','who','will','would',
        'could','should','can','may','just','also','more','after','before',
        'between','each','other','some','such','same','over','own'
    }
    return [t for t in text.split() if t not in stopwords and len(t) > 2]


class TFIDFVectorizer:
    def __init__(self, max_features=3000, ngram=True):
        self.max_features = max_features
        self.ngram = ngram
        self.vocab = {}
        self.idf = {}

    def _get_tokens(self, text):
        tokens = tokenize(text)
        if self.ngram:
            tokens += [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens

    def fit(self, texts):
        print("  Fitting TF-IDF vocabulary...")
        N = len(texts)
        df_counts = Counter()
        for text in texts:
            df_counts.update(set(self._get_tokens(text)))
        idf_all = {tok: math.log((N+1)/(cnt+1))+1 for tok, cnt in df_counts.items()}
        top_tokens = sorted(idf_all, key=idf_all.get, reverse=True)[:self.max_features]
        self.vocab = {tok: i for i, tok in enumerate(top_tokens)}
        self.idf   = {tok: idf_all[tok] for tok in top_tokens}
        print(f"  TF-IDF vocab size: {len(self.vocab):,}")

    def transform(self, texts):
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self._get_tokens(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            for tok, count in tf.items():
                if tok in self.vocab:
                    matrix[row, self.vocab[tok]] = (count / len(tokens)) * self.idf[tok]
        return matrix

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return [tok for tok, _ in sorted(self.vocab.items(), key=lambda x: x[1])]


class StandardScaler:
    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return (X - self.mean_) / self.std_


def build_features(df):
    print("\nExtracting features...")
    texts = df['article_text'].tolist()
    basic, basic_names = extract_basic_stats(df)
    basic_scaled = StandardScaler().fit_transform(basic)
    tfidf = TFIDFVectorizer(max_features=3000, ngram=True)
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")
    X = np.hstack([basic_scaled, tfidf_matrix])
    print(f"  Total features: {X.shape[1]:,}")
    return X, basic_names + tfidf.get_feature_names()


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    probs = np.bincount(y) / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_indices):
    """
    Find the best split across given features.
    KEY FIX: TF-IDF columns are sparse (80%+ zeros).
    np.percentile on a sparse column returns all zeros, making every
    threshold identical and useless. Instead we use the median of
    non-zero values, plus 0 itself as a threshold (word present vs absent).
    """
    best_gini, best_feat, best_thresh = float('inf'), None, None
    for feat_idx in feature_indices:
        col = X[:, feat_idx]
        nonzero = col[col > 0]
        if len(nonzero) == 0:
            continue  # all-zero column, skip entirely
        # Threshold 1: 0  (splits "word absent" vs "word present")
        # Threshold 2+: percentiles of non-zero values only
        thresholds = np.concatenate([[0.0], np.unique(np.percentile(nonzero, [50, 75]))])
        for thresh in thresholds:
            lm = col <= thresh
            if lm.sum() == 0 or (~lm).sum() == 0:
                continue
            n = len(y)
            g = (lm.sum()/n)*gini_impurity(y[lm]) + ((~lm).sum()/n)*gini_impurity(y[~lm])
            if g < best_gini:
                best_gini, best_feat, best_thresh = g, feat_idx, thresh
    return best_feat, best_thresh, best_gini


class Node:
    __slots__ = ['feature','threshold','left','right','prediction','gini']
    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.prediction = self.gini = None


class DecisionTree:
    def __init__(self, max_depth=15, min_samples_split=10, min_samples_leaf=5, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def _build(self, X, y, depth):
        node = Node()
        node.gini = gini_impurity(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            node.prediction = int(np.bincount(y).argmax())
            return node
        feat_idx = (np.random.choice(self.n_features_, self.max_features, replace=False)
                    if self.max_features and self.max_features < self.n_features_
                    else np.arange(self.n_features_))
        feat, thresh, _ = best_split(X, y, feat_idx)
        if feat is None:
            node.prediction = int(np.bincount(y).argmax())
            return node
        lm = X[:, feat] <= thresh
        if lm.sum() < self.min_samples_leaf or (~lm).sum() < self.min_samples_leaf:
            node.prediction = int(np.bincount(y).argmax())
            return node
        n = len(y)
        self.feature_importances_[feat] += (node.gini
            - (lm.sum()/n)*gini_impurity(y[lm])
            - ((~lm).sum()/n)*gini_impurity(y[~lm])) * n
        node.feature, node.threshold = feat, thresh
        node.left  = self._build(X[lm],  y[lm],  depth+1)
        node.right = self._build(X[~lm], y[~lm], depth+1)
        return node

    def predict(self, X):
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            node = self.root
            while node.prediction is None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = node.prediction
        return out


class RandomForest:
    def __init__(self, n_trees=50, max_depth=12, min_samples_split=10,
                 min_samples_leaf=5, max_features='sqrt', random_seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_str = max_features
        self.random_seed = random_seed
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        n_samples, n_features = X.shape
        max_feat = max(1, int(np.sqrt(n_features)) * 10) if self.max_features_str == 'sqrt' else n_features  # 5x sqrt for sparse TF-IDF
        self.feature_importances_ = np.zeros(n_features)
        self.trees = []

        for i in range(self.n_trees):
            if (i+1) % 10 == 0:
                print(f"    Training tree {i+1}/{self.n_trees}...")
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(self.max_depth, self.min_samples_split,
                                self.min_samples_leaf, max_feat)
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_

        self.feature_importances_ /= self.n_trees

    def predict(self, X):
        votes = np.zeros((len(X), 2), dtype=int)
        for tree in self.trees:
            preds = tree.predict(X)
            for i, p in enumerate(preds):
                votes[i, p] += 1
        return votes.argmax(axis=1)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    classes = sorted(np.unique(y_true).tolist())
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    return cm


def classification_report(y_true, y_pred):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print('-' * 55)
    f1s = []
    for label in labels:
        tp = np.sum((y_pred==label)&(y_true==label))
        fp = np.sum((y_pred==label)&(y_true!=label))
        fn = np.sum((y_pred!=label)&(y_true==label))
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f1s.append(f1)
        name = 'Not Propaganda' if label==0 else 'Propaganda'
        print(f"{name:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {np.sum(y_true==label):>10,}")
    macro = np.mean(f1s)
    print(f"\n{'Macro F1':<20} {macro:>10.4f}")
    return macro


def evaluate(model, X_test, y_test, name):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    macro = classification_report(y_test, y_pred)
    return y_pred, accuracy_score(y_test, y_pred), macro


def plot_results(dt, rf, X_test, y_test, feature_names):
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0F172A')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    DARK='#0F172A'; CARD='#1E293B'; BLUE='#38BDF8'
    PURPLE='#A78BFA'; GREEN='#4ADE80'; TEXT='#F1F5F9'; MUTED='#94A3B8'

    def style_ax(ax, title, color):
        ax.set_facecolor(CARD)
        for s in ax.spines.values(): s.set_edgecolor('#334155')
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title, color=color, fontsize=12, fontweight='bold', pad=10)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

    def get_metrics(yt, yp):
        acc = accuracy_score(yt, yp)
        f1s = []
        for lbl in [0,1]:
            tp=np.sum((yp==lbl)&(yt==lbl)); fp=np.sum((yp==lbl)&(yt!=lbl)); fn=np.sum((yp!=lbl)&(yt==lbl))
            p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
            f1s.append(2*p*r/(p+r) if p+r>0 else 0)
        return acc, f1s[0], f1s[1], np.mean(f1s)

    ax0 = fig.add_subplot(gs[0,:])
    style_ax(ax0, 'Decision Tree vs Random Forest — Performance Metrics', BLUE)
    m_dt = get_metrics(y_test, y_pred_dt)
    m_rf = get_metrics(y_test, y_pred_rf)
    xlabels = ['Accuracy','F1 (Not Prop)','F1 (Propaganda)','Macro F1']
    x = np.arange(4); w = 0.35
    b_dt = ax0.bar(x-w/2, m_dt, w, label='Decision Tree', color=PURPLE, alpha=0.85)
    b_rf = ax0.bar(x+w/2, m_rf, w, label='Random Forest', color=GREEN,  alpha=0.85)
    for bar,c in [(b,PURPLE) for b in b_dt]+[(b,GREEN) for b in b_rf]:
        ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', color=c, fontsize=9, fontweight='bold')
    ax0.set_xticks(x); ax0.set_xticklabels(xlabels, color=TEXT, fontsize=10)
    ax0.set_ylim(0,1.15); ax0.set_ylabel('Score',color=MUTED)
    ax0.legend(facecolor=CARD, edgecolor='#334155', labelcolor=TEXT, fontsize=10)
    ax0.axhline(1.0, color='#334155', linestyle='--', linewidth=0.8)

    for ci,(name,yp,col) in enumerate([('Decision Tree',y_pred_dt,PURPLE),('Random Forest',y_pred_rf,GREEN)]):
        ax = fig.add_subplot(gs[1,ci]); style_ax(ax, f'{name} — Confusion Matrix', col)
        cm = confusion_matrix(y_test, yp); cn = cm.astype(float)/cm.sum(axis=1)[:,None]
        ax.imshow(cn, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Not Prop','Propaganda'],color=TEXT)
        ax.set_yticklabels(['Not Prop','Propaganda'],color=TEXT)
        ax.set_xlabel('Predicted',color=MUTED); ax.set_ylabel('Actual',color=MUTED)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{cm[i,j]:,}\n({cn[i,j]:.2%})',ha='center',va='center',
                        fontsize=10,fontweight='bold',color='white' if cn[i,j]>0.5 else '#1E293B')

    top_n = 20
    for ci,(name,model,col) in enumerate([('Decision Tree',dt,PURPLE),('Random Forest',rf,GREEN)]):
        ax = fig.add_subplot(gs[2,ci]); style_ax(ax, f'{name} — Top {top_n} Features', col)
        imp = model.feature_importances_
        ti  = np.argsort(imp)[-top_n:][::-1]
        ax.barh(np.arange(top_n), imp[ti][::-1], color=col, alpha=0.8)
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels([feature_names[i] if i<len(feature_names) else f'f{i}' for i in ti][::-1],
                           color=TEXT, fontsize=8)
        ax.set_xlabel('Importance Score', color=MUTED)

    fig.text(0.5,0.98,'QProp — Decision Tree vs Random Forest (From Scratch)',
             ha='center',color=TEXT,fontsize=14,fontweight='bold')
    plt.savefig('qprop_model_results.png', dpi=150, bbox_inches='tight', facecolor=DARK)
    print("\nSaved → qprop_model_results.png")


def main():
    df = load_data('proppy_1.0.train.tsv')
    X, feature_names = build_features(df)
    y = df['propaganda_label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    print("\nTraining Decision Tree...")
    dt = DecisionTree(max_depth=15, min_samples_split=10, min_samples_leaf=5)
    dt.fit(X_train, y_train)

    print("\nTraining Random Forest (100 trees)...")
    rf = RandomForest(n_trees=100, max_depth=15, min_samples_split=10,
                      min_samples_leaf=5, max_features='sqrt', random_seed=42)
    rf.fit(X_train, y_train)

    _, acc_dt, f1_dt = evaluate(dt, X_test, y_test, 'DECISION TREE')
    _, acc_rf, f1_rf = evaluate(rf, X_test, y_test, 'RANDOM FOREST')

    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'-'*44}")
    print(f"{'Decision Tree':<22} {acc_dt:>10.4f} {f1_dt:>10.4f}")
    print(f"{'Random Forest':<22} {acc_rf:>10.4f} {f1_rf:>10.4f}")
    print(f"{'Improvement':<22} {acc_rf-acc_dt:>+10.4f} {f1_rf-f1_dt:>+10.4f}")

    plot_results(dt, rf, X_test, y_test, feature_names)

if __name__ == '__main__':
    main()