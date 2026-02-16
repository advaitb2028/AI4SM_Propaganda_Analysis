"""
QProp Dataset Preprocessing Script with Character Tri-grams
Dataset: Proppy Corpus 1.0 - Propaganda Detection Dataset
Source: https://zenodo.org/records/3271522

This script processes the proppy_1.0.train.tsv file with:
- Statistical analysis and visualizations
- Character tri-gram feature extraction
- Social media feature extraction
- Advanced text preprocessing
- ML-ready feature vectors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import json
import string

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")


class SocialMediaFeatureExtractor:
    """Extract social media-specific features before text cleaning"""
    
    @staticmethod
    def extract_hashtags(text: str) -> list:
        """Extract all hashtags from text"""
        return re.findall(r'#\w+', str(text))
    
    @staticmethod
    def extract_mentions(text: str) -> list:
        """Extract all mentions (@username) from text"""
        return re.findall(r'@\w+', str(text))
    
    @staticmethod
    def extract_urls(text: str) -> list:
        """Extract all URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, str(text))
    
    @staticmethod
    def count_exclamation(text: str) -> int:
        """Count exclamation marks (common in propaganda)"""
        return str(text).count('!')
    
    @staticmethod
    def count_question(text: str) -> int:
        """Count question marks"""
        return str(text).count('?')
    
    @staticmethod
    def count_all_caps_words(text: str) -> int:
        """Count words in ALL CAPS (common in propaganda)"""
        words = str(text).split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    def extract_all_features(self, text: str) -> dict:
        """Extract all stylistic features (no emojis)"""
        text_str = str(text)
        return {
            'hashtag_count': len(self.extract_hashtags(text_str)),
            'mention_count': len(self.extract_mentions(text_str)),
            'url_count': len(self.extract_urls(text_str)),
            'exclamation_count': self.count_exclamation(text_str),
            'question_count': self.count_question(text_str),
            'all_caps_word_count': self.count_all_caps_words(text_str),
            'uppercase_ratio': sum(1 for c in text_str if c.isupper()) / len(text_str) if len(text_str) > 0 else 0,
            'punctuation_count': sum(1 for c in text_str if c in string.punctuation)
        }


class CharTrigramTokenizer:
    """
    Character Tri-gram Tokenizer for Propaganda Detection
    
    Character tri-grams capture:
    - Spelling patterns
    - Punctuation usage patterns (!!!!, ???)
    - Stylistic features
    - Misspellings and intentional distortions
    """
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocabulary = {}
        self.char_trigram_counts = Counter()
        
    def extract_char_trigrams_from_words(self, text: str) -> list:
        """
        Extract character tri-grams word by word
        This captures word boundaries better
        
        Example: "fake" -> ["<fa", "fak", "ake", "ke>"]
        """
        words = str(text).split()
        all_trigrams = []
        
        for word in words:
            if len(word) < 2:
                continue
            marked_word = f"<{word}>"
            
            for i in range(len(marked_word) - 2):
                trigram = marked_word[i:i+3]
                all_trigrams.append(trigram)
        
        return all_trigrams
    
    def fit(self, texts: list):
        """Build vocabulary from texts"""
        print(f"\nFitting character tri-gram tokenizer on {len(texts)} documents...")
        
        # Count all tri-grams
        for text in texts:
            trigrams = self.extract_char_trigrams_from_words(text)
            self.char_trigram_counts.update(trigrams)
        
        # Build vocabulary (most common tri-grams)
        if self.max_features:
            most_common = self.char_trigram_counts.most_common(self.max_features)
        else:
            most_common = self.char_trigram_counts.most_common()
        
        self.vocabulary = {trigram: idx for idx, (trigram, _) in enumerate(most_common)}
        
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        print(f"  Total unique tri-grams: {len(self.char_trigram_counts)}")
        print(f"  Most common tri-grams: {self.char_trigram_counts.most_common(10)}")
    
    def transform(self, texts: list) -> np.ndarray:
        """Transform texts to tri-gram frequency vectors"""
        vectors = []
        
        for text in texts:
            trigrams = self.extract_char_trigrams_from_words(text)
            
            # Create frequency vector
            vector = np.zeros(len(self.vocabulary))
            trigram_counts = Counter(trigrams)
            
            for trigram, count in trigram_counts.items():
                if trigram in self.vocabulary:
                    vector[self.vocabulary[trigram]] = count
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """Fit vocabulary and transform texts in one step"""
        self.fit(texts)
        return self.transform(texts)


def load_data(filepath):
    """Load the QProp dataset from TSV file."""
    print("Loading data...")
    
    # Column names based on dataset documentation
    columns = [
        'article_text',
        'event_location', 
        'average_tone',
        'article_date',
        'article_ID',
        'article_URL',
        'MBFC_factuality_label',
        'article_URL_dup',  
        'MBFC_factuality_label_dup',
        'URL_to_MBFC_page',
        'source_name',
        'MBFC_notes_about_source',
        'MBFC_bias_label',
        'source_URL',
        'propaganda_label'
    ]
    
    df = pd.read_csv(filepath, sep='\t', names=columns, header=None, 
                     low_memory=False, encoding='utf-8')
    
    print(f"Loaded {len(df):,} articles")
    return df


def clean_data(df):
    """Perform initial data cleaning."""
    print("\nCleaning data...")
    
    # Remove duplicate columns
    df = df.drop(columns=['article_URL_dup', 'MBFC_factuality_label_dup'], errors='ignore')
    
    # Convert date column
    df['article_date'] = pd.to_datetime(df['article_date'], errors='coerce')
    
    # Convert tone to numeric
    df['average_tone'] = pd.to_numeric(df['average_tone'], errors='coerce')
    
    # Clean text columns
    text_cols = ['article_text', 'source_name', 'MBFC_notes_about_source']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Convert propaganda label to binary
    df['propaganda_label'] = df['propaganda_label'].astype(int)
    
    # Check if labels are -1/1 format and convert to 0/1 
    unique_labels = df['propaganda_label'].unique()
    if -1 in unique_labels:
        print("  Note: Converting labels from -1/1 format to 0/1 format")
        print(f"  Original: -1={sum(df['propaganda_label']==-1)}, 1={sum(df['propaganda_label']==1)}")
        df['propaganda_label'] = df['propaganda_label'].replace(-1, 0)
        print(f"  Converted: 0={sum(df['propaganda_label']==0)}, 1={sum(df['propaganda_label']==1)}")
    
    return df


def compute_text_statistics(df):
    """Compute basic text-level statistics."""
    print("\nComputing basic text statistics...")
    
    # Word count
    df['word_count'] = df['article_text'].apply(lambda x: len(str(x).split()))
    
    # Character count
    df['char_count'] = df['article_text'].apply(lambda x: len(str(x)))
    
    # Sentence count
    df['sentence_count'] = df['article_text'].apply(
        lambda x: len(re.findall(r'[.!?]+', str(x)))
    )
    
    # Average word length
    df['avg_word_length'] = df['article_text'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    return df


def extract_advanced_features(df):
    """Extract advanced social media and stylistic features."""
    print("\nExtracting advanced features (hashtags, URLs, caps, punctuation)...")
    
    extractor = SocialMediaFeatureExtractor()
    
    social_features = df['article_text'].apply(extractor.extract_all_features)
    
    # Convert to DataFrame
    social_features_df = pd.DataFrame(social_features.tolist())
    
    # Add to main dataframe
    for col in social_features_df.columns:
        df[col] = social_features_df[col]
    
    print(f"  Extracted {len(social_features_df.columns)} advanced features")
    
    return df


def extract_character_trigrams(df, max_features=5000):
    """
    Extract character tri-gram features for machine learning.
    Returns both the feature matrix and the tokenizer for later use.
    """
    print("\n" + "="*60)
    print("CHARACTER TRI-GRAM FEATURE EXTRACTION")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = CharTrigramTokenizer(max_features=max_features)
    
    # Fit and transform on article texts
    texts = df['article_text'].tolist()
    trigram_matrix = tokenizer.fit_transform(texts)
    
    print(f"\nTri-gram feature matrix shape: {trigram_matrix.shape}")
    print(f"Sparsity: {(trigram_matrix == 0).sum() / trigram_matrix.size * 100:.2f}%")
    
    return trigram_matrix, tokenizer


def generate_summary_statistics(df):
    """Generate comprehensive summary statistics."""
    print("\nGenerating summary statistics...")
    
    stats = {}
    
    # Basic dataset info
    stats['total_articles'] = len(df)
    stats['date_range'] = {
        'earliest': df['article_date'].min(),
        'latest': df['article_date'].max()
    }
    
    # Propaganda distribution
    propaganda_dist = df['propaganda_label'].value_counts()
    stats['propaganda_distribution'] = {
        'propaganda': int(propaganda_dist.get(1, 0)),
        'non_propaganda': int(propaganda_dist.get(0, 0)),
        'propaganda_percentage': float(propaganda_dist.get(1, 0) / len(df) * 100)
    }
    
    # Source statistics
    stats['unique_sources'] = int(df['source_name'].nunique())
    stats['top_sources'] = df['source_name'].value_counts().head(10).to_dict()
    
    # Factuality label distribution
    if 'MBFC_factuality_label' in df.columns:
        stats['factuality_distribution'] = df['MBFC_factuality_label'].value_counts().to_dict()
    
    # Bias label distribution
    if 'MBFC_bias_label' in df.columns:
        stats['bias_distribution'] = df['MBFC_bias_label'].value_counts().to_dict()
    
    # Text statistics
    stats['text_statistics'] = {
        'avg_word_count': float(df['word_count'].mean()),
        'median_word_count': float(df['word_count'].median()),
        'avg_char_count': float(df['char_count'].mean()),
        'avg_sentences': float(df['sentence_count'].mean())
    }
    
    # Tone statistics
    stats['tone_statistics'] = {
        'mean': float(df['average_tone'].mean()),
        'median': float(df['average_tone'].median()),
        'std': float(df['average_tone'].std())
    }
    
    # Advanced feature statistics
    if 'exclamation_count' in df.columns:
        stats['stylistic_features'] = {
            'avg_exclamations': float(df['exclamation_count'].mean()),
            'avg_questions': float(df['question_count'].mean()),
            'avg_caps_words': float(df['all_caps_word_count'].mean()),
            'avg_hashtags': float(df['hashtag_count'].mean()),
            'avg_mentions': float(df['mention_count'].mean()),
            'avg_urls': float(df['url_count'].mean())
        }
    
    # Geographic distribution (if available)
    if 'event_location' in df.columns:
        location_counts = df['event_location'].value_counts().head(20)
        stats['top_locations'] = location_counts.to_dict()
    
    return stats


def analyze_propaganda_patterns(df):
    """Analyze patterns in propaganda vs non-propaganda articles."""
    print("\nAnalyzing propaganda patterns...")
    
    patterns = {}
    
    # Text length comparison
    patterns['text_length_by_label'] = df.groupby('propaganda_label')['word_count'].describe().to_dict()
    
    # Tone comparison
    patterns['tone_by_label'] = df.groupby('propaganda_label')['average_tone'].describe().to_dict()
    
    # Stylistic features comparison
    if 'exclamation_count' in df.columns:
        patterns['exclamations_by_label'] = df.groupby('propaganda_label')['exclamation_count'].describe().to_dict()
        patterns['caps_words_by_label'] = df.groupby('propaganda_label')['all_caps_word_count'].describe().to_dict()
    
    # Factuality vs propaganda
    if 'MBFC_factuality_label' in df.columns:
        crosstab = pd.crosstab(df['MBFC_factuality_label'], df['propaganda_label'], normalize='index')
        patterns['factuality_vs_propaganda'] = crosstab.to_dict()
    
    # Bias vs propaganda
    if 'MBFC_bias_label' in df.columns:
        crosstab = pd.crosstab(df['MBFC_bias_label'], df['propaganda_label'], normalize='index')
        patterns['bias_vs_propaganda'] = crosstab.to_dict()
    
    # Top propaganda sources
    propaganda_articles = df[df['propaganda_label'] == 1]
    patterns['top_propaganda_sources'] = propaganda_articles['source_name'].value_counts().head(20).to_dict()
    
    # Top non-propaganda sources
    non_propaganda_articles = df[df['propaganda_label'] == 0]
    patterns['top_non_propaganda_sources'] = non_propaganda_articles['source_name'].value_counts().head(20).to_dict()
    
    return patterns


def temporal_analysis(df):
    """Analyze temporal patterns."""
    print("\nPerforming temporal analysis...")
    
    temporal = {}
    
    # Articles per month
    df['year_month'] = df['article_date'].dt.to_period('M')
    monthly_counts = df.groupby('year_month').size()
    # Convert Period index to string for JSON serialization
    temporal['articles_per_month'] = {str(k): int(v) for k, v in monthly_counts.items()}
    
    # Propaganda ratio over time
    monthly_propaganda = df.groupby(['year_month', 'propaganda_label']).size().unstack(fill_value=0)
    if 1 in monthly_propaganda.columns and 0 in monthly_propaganda.columns:
        propaganda_ratio = monthly_propaganda[1] / (monthly_propaganda[0] + monthly_propaganda[1])
        # Convert Period index to string for JSON serialization
        temporal['propaganda_ratio_over_time'] = {str(k): float(v) for k, v in propaganda_ratio.items()}
    
    return temporal


def create_visualizations(df, output_dir='./'):
    """Create visualization plots."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QProp Dataset Analysis with Advanced Features', fontsize=16, fontweight='bold')
    
    # 1. Propaganda distribution
    propaganda_counts = df['propaganda_label'].value_counts()
    axes[0, 0].bar(['Non-Propaganda', 'Propaganda'], 
                   [propaganda_counts.get(0, 0), propaganda_counts.get(1, 0)],
                   color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Propaganda Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Word count distribution
    axes[0, 1].hist([df[df['propaganda_label']==0]['word_count'], 
                     df[df['propaganda_label']==1]['word_count']], 
                    bins=50, label=['Non-Propaganda', 'Propaganda'], alpha=0.7)
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 3000)
    
    # 3. Exclamation marks (propaganda indicator)
    if 'exclamation_count' in df.columns:
        axes[0, 2].hist([df[df['propaganda_label']==0]['exclamation_count'], 
                         df[df['propaganda_label']==1]['exclamation_count']], 
                        bins=30, label=['Non-Propaganda', 'Propaganda'], alpha=0.7)
        axes[0, 2].set_title('Exclamation Mark Usage')
        axes[0, 2].set_xlabel('Exclamation Count')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].set_xlim(0, 20)
    
    # 4. Top sources
    top_sources = df['source_name'].value_counts().head(15)
    axes[1, 0].barh(range(len(top_sources)), top_sources.values)
    axes[1, 0].set_yticks(range(len(top_sources)))
    axes[1, 0].set_yticklabels(top_sources.index, fontsize=8)
    axes[1, 0].set_title('Top 15 Sources')
    axes[1, 0].set_xlabel('Article Count')
    axes[1, 0].invert_yaxis()
    
    # 5. Articles over time
    df['year_month'] = df['article_date'].dt.to_period('M')
    monthly = df.groupby('year_month').size()
    axes[1, 1].plot(range(len(monthly)), monthly.values)
    axes[1, 1].set_title('Articles Over Time')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Article Count')
    
    # 6. ALL CAPS words (propaganda indicator)
    if 'all_caps_word_count' in df.columns:
        axes[1, 2].hist([df[df['propaganda_label']==0]['all_caps_word_count'], 
                         df[df['propaganda_label']==1]['all_caps_word_count']], 
                        bins=20, label=['Non-Propaganda', 'Propaganda'], alpha=0.7)
        axes[1, 2].set_title('ALL CAPS Words Usage')
        axes[1, 2].set_xlabel('ALL CAPS Word Count')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/qprop_analysis_enhanced.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/qprop_analysis_enhanced.png")
    
    return fig


def export_cleaned_data(df, output_dir='./'):
    """Export cleaned dataset."""
    print("\nExporting cleaned data...")
    
    # Select key columns for export
    export_cols = [
        'article_ID', 'article_date', 'source_name', 'article_URL',
        'propaganda_label', 'MBFC_factuality_label', 'MBFC_bias_label',
        'word_count', 'char_count', 'sentence_count', 'average_tone',
        'hashtag_count', 'mention_count', 'url_count', 
        'exclamation_count', 'question_count', 'all_caps_word_count',
        'uppercase_ratio', 'punctuation_count',
        'event_location', 'article_text'
    ]
    
    # Filter to existing columns
    export_cols = [col for col in export_cols if col in df.columns]
    
    df_export = df[export_cols].copy()
    
    # Export to CSV
    output_file = f'{output_dir}/qprop_cleaned_enhanced.csv'
    df_export.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved cleaned data to {output_file}")
    
    # Export summary CSV with aggregated stats
    agg_dict = {
        'article_ID': 'count',
        'propaganda_label': 'mean',
        'word_count': 'mean',
        'average_tone': 'mean'
    }
    
    if 'exclamation_count' in df.columns:
        agg_dict['exclamation_count'] = 'mean'
        agg_dict['all_caps_word_count'] = 'mean'
    
    summary_df = df.groupby('source_name').agg(agg_dict).round(3)
    summary_df.columns = ['article_count', 'propaganda_ratio', 'avg_words', 'avg_tone'] + \
                         (['avg_exclamations', 'avg_caps_words'] if 'exclamation_count' in df.columns else [])
    summary_df = summary_df.sort_values('article_count', ascending=False)
    summary_df.to_csv(f'{output_dir}/source_summary_enhanced.csv')
    print(f"Saved source summary to {output_dir}/source_summary_enhanced.csv")


def export_ml_features(df, trigram_matrix, trigram_tokenizer, output_dir='./'):
    """
    Export ML-ready features for model training.
    """
    print("\n" + "="*60)
    print("EXPORTING ML-READY FEATURES")
    print("="*60)
    
    # Get basic features
    basic_feature_cols = [
        'word_count', 'char_count', 'sentence_count', 'avg_word_length',
        'average_tone', 'hashtag_count', 'mention_count', 'url_count',
        'exclamation_count', 'question_count', 'all_caps_word_count',
        'uppercase_ratio', 'punctuation_count'
    ]
    
    # Filter to existing columns
    basic_feature_cols = [col for col in basic_feature_cols if col in df.columns]
    basic_features = df[basic_feature_cols].values
    
    # Combine basic features with tri-gram features
    combined_features = np.hstack([basic_features, trigram_matrix])
    
    # Get labels
    labels = df['propaganda_label'].values
    
    # Save features and labels
    np.save(f'{output_dir}/qprop_features_combined.npy', combined_features)
    np.save(f'{output_dir}/qprop_features_trigrams_only.npy', trigram_matrix)
    np.save(f'{output_dir}/qprop_features_basic_only.npy', basic_features)
    np.save(f'{output_dir}/qprop_labels.npy', labels)
    
    # Save feature names
    feature_names = {
        'basic_features': basic_feature_cols,
        'trigram_vocab_size': len(trigram_tokenizer.vocabulary),
        'total_features': combined_features.shape[1]
    }
    
    with open(f'{output_dir}/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save tri-gram vocabulary
    with open(f'{output_dir}/char_trigram_vocab.json', 'w') as f:
        json.dump(trigram_tokenizer.vocabulary, f, indent=2)
    
    print(f"\nML Features saved:")
    print(f"  - Combined features: {combined_features.shape}")
    print(f"  - Basic features only: {basic_features.shape}")
    print(f"  - Tri-gram features only: {trigram_matrix.shape}")
    print(f"  - Labels: {labels.shape}")
    print(f"  - Feature names and vocabulary saved to JSON files")


def main():
    """Main preprocessing pipeline with character tri-grams."""
    print("="*60)
    print("QProp Dataset Preprocessing Pipeline (Enhanced)")
    print("With Character Tri-grams and Advanced Features")
    print("="*60)
    
    # Configuration
    input_file = 'proppy_1.0.train.tsv'
    output_dir = './'
    max_trigram_features = 5000  
    
    # Load and clean data
    df = load_data(input_file)
    df = clean_data(df)
    
    # Compute basic text statistics
    df = compute_text_statistics(df)
    
    # Extract advanced features (hashtags, URLs, exclamations, ALL CAPS, etc.)
    df = extract_advanced_features(df)
    
    # Extract character tri-gram features
    trigram_matrix, trigram_tokenizer = extract_character_trigrams(df, max_features=max_trigram_features)
    
    # Generate statistical analyses
    summary_stats = generate_summary_statistics(df)
    propaganda_patterns = analyze_propaganda_patterns(df)
    temporal = temporal_analysis(df)
    
    # Save statistics to JSON
    all_stats = {
        'summary': summary_stats,
        'propaganda_patterns': propaganda_patterns,
        'temporal': temporal
    }
    
    with open(f'{output_dir}/qprop_statistics_enhanced.json', 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print("\nSaved statistics to qprop_statistics_enhanced.json")
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Export cleaned data
    export_cleaned_data(df, output_dir)
    
    # Export ML-ready features
    export_ml_features(df, trigram_matrix, trigram_tokenizer, output_dir)
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal Articles: {summary_stats['total_articles']:,}")
    print(f"Date Range: {summary_stats['date_range']['earliest']} to {summary_stats['date_range']['latest']}")
    print(f"\nPropaganda Distribution:")
    print(f"  - Propaganda: {summary_stats['propaganda_distribution']['propaganda']:,} ({summary_stats['propaganda_distribution']['propaganda_percentage']:.1f}%)")
    print(f"  - Non-Propaganda: {summary_stats['propaganda_distribution']['non_propaganda']:,}")
    print(f"\nUnique Sources: {summary_stats['unique_sources']}")
    print(f"\nText Statistics:")
    print(f"  - Avg Word Count: {summary_stats['text_statistics']['avg_word_count']:.0f}")
    print(f"  - Median Word Count: {summary_stats['text_statistics']['median_word_count']:.0f}")
    
    if 'stylistic_features' in summary_stats:
        print(f"\nStylistic Features (Propaganda Indicators):")
        print(f"  - Avg Exclamations: {summary_stats['stylistic_features']['avg_exclamations']:.2f}")
        print(f"  - Avg ALL CAPS Words: {summary_stats['stylistic_features']['avg_caps_words']:.2f}")
        print(f"  - Avg Hashtags: {summary_stats['stylistic_features']['avg_hashtags']:.2f}")
    
    print(f"\nFeature Extraction:")
    print(f"  - Character tri-grams extracted: {trigram_matrix.shape[1]:,}")
    print(f"  - Total features for ML: {trigram_matrix.shape[1] + len([c for c in df.columns if c in ['word_count', 'exclamation_count', 'all_caps_word_count']])}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  qprop_analysis_enhanced.png - Visualizations")
    print("  qprop_statistics_enhanced.json - Statistics")
    print("  qprop_cleaned_enhanced.csv - Cleaned dataset")
    print("  source_summary_enhanced.csv - Source-level stats")
    print("  qprop_features_combined.npy - ML features (basic + trigrams)")
    print("  qprop_features_trigrams_only.npy - Trigram features only")
    print("  qprop_features_basic_only.npy - Basic features only")
    print("  qprop_labels.npy - Labels for ML")
    print("  char_trigram_vocab.json - Trigram vocabulary")
    print("  feature_names.json - Feature documentation")
    print("="*60)


if __name__ == "__main__":
    main()