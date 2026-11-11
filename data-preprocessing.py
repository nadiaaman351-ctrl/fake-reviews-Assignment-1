"""Member 2 – NLP Preprocessing 
Responsibilities:

Clean the review text.

Remove punctuation and special characters.

Apply tokenization.

Remove stopwords.

Apply stemming or lemmatization.

Apply TF-IDF vectorization.
Deliverables:

Preprocessed dataset.

Python preprocessing code.

Before-and-after text examples."""

import pandas as pd
import numpy as np
import re
import html
import joblib

from pathlib import Path
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "fake reviews dataset.csv"

OUTPUT_DIR = BASE_DIR / "MuhammadDaud_0855_preprocessing_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading Dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place fake reviews dataset.csv in the same folder "
        "as this Python file."
    )

df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

print("\nDataset Loaded Successfully!")

print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())


# ==========================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = ["text_", "label"]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing."
        )


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print(df[required_columns].isnull().sum())

# Remove records with missing review text or label
df = df.dropna(subset=["text_", "label"]).copy()

df["text_"] = df["text_"].astype(str)

print("\nMissing values handled successfully.")

print("Remaining Records:", len(df))


# ==========================================
# 5. INITIALIZE STEMMER
# ==========================================

stemmer = PorterStemmer()

# Load English stopwords from scikit-learn.
# Preserve negation words because they can
# change the meaning of product reviews.

stop_words = set(ENGLISH_STOP_WORDS)

stop_words = stop_words - {"not", "no", "nor"}


# ==========================================
# 6. TEXT CLEANING
# ==========================================

def clean_text(text):

    # Convert HTML entities
    text = html.unescape(str(text))

    # Convert text into lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove numbers and special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 7. TOKENIZATION
# ==========================================

def tokenize_text(text):

    # Split sentence into individual words
    tokens = text.split()

    return tokens


# ==========================================
# 8. STOPWORD REMOVAL
# ==========================================

def remove_stopwords(tokens):

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return filtered_tokens


# ==========================================
# 9. STEMMING
# ==========================================

def apply_stemming(tokens):

    stemmed_tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return stemmed_tokens


# ==========================================
# 10. COMPLETE PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    # Step 1: Text cleaning
    cleaned_text = clean_text(text)

    # Step 2: Tokenization
    tokens = tokenize_text(cleaned_text)

    # Step 3: Stopword removal
    filtered_tokens = remove_stopwords(tokens)

    # Step 4: Stemming
    stemmed_tokens = apply_stemming(filtered_tokens)

    # Convert words back into a sentence
    final_text = " ".join(stemmed_tokens)

    return final_text


# ==========================================
# 11. APPLY PREPROCESSING TO DATASET
# ==========================================

print("\n========== TEXT PREPROCESSING ==========")

print("Cleaning reviews...")

df["cleaned_text"] = df["text_"].apply(preprocess_text)

print("Text cleaning completed!")

print("Tokenization completed!")

print("Stopword removal completed!")

print("Stemming completed!")


# ==========================================
# 12. BEFORE AND AFTER EXAMPLES
# ==========================================

print("\n========== BEFORE AND AFTER EXAMPLES ==========")

for i in range(min(5, len(df))):

    print("\nOriginal Review:")
    print(df["text_"].iloc[i])

    print("\nCleaned Review:")
    print(df["cleaned_text"].iloc[i])

    print("-" * 60)


# ==========================================
# 13. SAVE PREPROCESSED DATASET
# ==========================================

print("\nSaving Preprocessed Dataset...")

preprocessed_file = OUTPUT_DIR / "preprocessed_reviews.csv"

df.to_csv(
    preprocessed_file,
    index=False
)

print("Preprocessed dataset saved!")


# ==========================================
# 14. PREPARE FEATURES AND LABELS
# ==========================================

X = df["cleaned_text"]

y = df["label"]


# ==========================================
# 15. TRAIN AND TEST SPLIT
# ==========================================

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))


# ==========================================
# 16. TF-IDF FEATURE EXTRACTION
# ==========================================

print("\n========== TF-IDF FEATURE EXTRACTION ==========")

tfidf = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95

)

# Fit vectorizer only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data using the same vectorizer
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Extraction Completed!")

print("Training Matrix Shape:", X_train_tfidf.shape)

print("Testing Matrix Shape:", X_test_tfidf.shape)

print("Total Features:", len(tfidf.get_feature_names_out()))


# ==========================================
# 17. SAVE TF-IDF FEATURES
# ==========================================

print("\nSaving TF-IDF Features...")

save_npz(
    OUTPUT_DIR / "X_train_tfidf.npz",
    X_train_tfidf
)

save_npz(
    OUTPUT_DIR / "X_test_tfidf.npz",
    X_test_tfidf
)


# ==========================================
# 18. SAVE TRAINING AND TESTING LABELS
# ==========================================

y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# ==========================================
# 19. SAVE TF-IDF VECTORIZER
# ==========================================

joblib.dump(
    tfidf,
    OUTPUT_DIR / "tfidf_vectorizer.pkl"
)


# ==========================================
# 20. SAVE FEATURE NAMES
# ==========================================

feature_names = tfidf.get_feature_names_out()

pd.DataFrame({
    "feature": feature_names
}).to_csv(
    OUTPUT_DIR / "tfidf_feature_names.csv",
    index=False
)


# ==========================================
# 21. FINAL RESULTS
# ==========================================

print("\n========== FINAL RESULTS ==========")

print("Total Reviews:", len(df))

print("Training Reviews:", len(X_train))

print("Testing Reviews:", len(X_test))

print("TF-IDF Features:", X_train_tfidf.shape[1])

print("Training Matrix:", X_train_tfidf.shape)

print("Testing Matrix:", X_test_tfidf.shape)

print("\nAll files saved successfully!")

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nMUHAMMAD DAUD (0855) - PREPROCESSING COMPLETED!")
