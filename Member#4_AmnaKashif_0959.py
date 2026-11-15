"""
=========================================================
MEMBER 4: AmnaKashif_0959
=========================================================

WORK DESCRIPTION:

Member 4 is responsible for saving, loading, and integrating
the trained models created by the other group members.

Her main tasks are:

1. Load the saved TF-IDF vectorizer created by Member 1.
2. Load the Word2Vec CBOW model created by Member 2.
3. Load the Word2Vec Skip-gram model created by Member 2.
4. Load the FastText model created by Member 3.
5. Create a function that accepts a new customer review.
6. Convert the review into numerical vectors using:
   - TF-IDF
   - Word2Vec CBOW
   - Word2Vec Skip-gram
   - FastText
7. Display the shape and numerical values of the generated vectors.

This allows the project to process new reviews using the
trained NLP representations.
8. Making the Report for Assignment #2

NOTE:
This code does not train the models or classify reviews as
fake or genuine. It only loads the existing models and
converts new reviews into numerical representations.
=========================================================
"""

import re
import pickle
import numpy as np

from pathlib import Path
from gensim.models import Word2Vec, FastText


# Project directory
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"


# Load trained models
def load_models():

    models = {}

    tfidf_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    if tfidf_path.exists():
        with open(tfidf_path, "rb") as file:
            models["tfidf"] = pickle.load(file)

    cbow_path = MODEL_DIR / "cbow.model"

    if cbow_path.exists():
        models["cbow"] = Word2Vec.load(str(cbow_path))

    skipgram_path = MODEL_DIR / "skipgram.model"

    if skipgram_path.exists():
        models["skipgram"] = Word2Vec.load(str(skipgram_path))

    fasttext_path = MODEL_DIR / "fasttext.model"

    if fasttext_path.exists():
        models["fasttext"] = FastText.load(str(fasttext_path))

    return models


# Basic tokenization
def tokenize(text):

    text = str(text).lower()

    words = re.findall(r"\b[a-z0-9]+\b", text)

    return words


# Convert document into vector
def vectorize_document(text, representation="tfidf"):

    models = load_models()

    if representation not in models:
        raise ValueError(
            f"Model '{representation}' not found. "
            f"Available models: {list(models.keys())}"
        )

    # TF-IDF
    if representation == "tfidf":

        vectorizer = models["tfidf"]

        vector = vectorizer.transform([text])

        return vector

    # Word2Vec and FastText
    model = models[representation]

    words = tokenize(text)

    vectors = []

    for word in words:

        try:
            vectors.append(model.wv[word])

        except KeyError:
            continue

    if len(vectors) == 0:

        return np.zeros(model.vector_size)

    document_vector = np.mean(vectors, axis=0)

    return document_vector


# Test all representations
if __name__ == "__main__":

    sample_text = "This product has excellent quality"

    models = load_models()

    print("Available Models:")

    for name in models:
        print(name)

    print("\nTesting Document Vectorization")

    for representation in models:

        vector = vectorize_document(
            sample_text,
            representation
        )

        print("\nRepresentation:", representation)

        if representation == "tfidf":
            print("Vector Shape:", vector.shape)
        else:
            print("Vector Shape:", vector.shape)

        print("Vector:", vector)