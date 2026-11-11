"""
Fake Reviews Detection:
Member 1 – Dataset Collection (Part 1)
Responsibilities:
- Find a suitable fake product review dataset.
- Download and understand the dataset.
- Identify fake and genuine review labels.
- Check missing values and duplicate records.
- Perform initial dataset analysis.
  Deliverables:
- Dataset CSV file.
- Dataset description.
- Number of fake and genuine reviews."""
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ==========================================
# 1. DATASET LOCATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "fake reviews dataset.csv"

OUTPUT_DIR = BASE_DIR / "AtizazAhmed_0839_eda_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

print("\n========== DATASET OVERVIEW ==========")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nFirst Five Records:")
print(df.head())

# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# ==========================================
# 4. DUPLICATE RECORDS
# ==========================================

print("\n========== DUPLICATE RECORDS ==========")

print("Duplicate Rows:", df.duplicated().sum())

print("Duplicate Review Text:",
      df["text_"].duplicated().sum())

# ==========================================
# 5. LABEL DISTRIBUTION
# ==========================================

print("\n========== LABEL DISTRIBUTION ==========")

label_counts = df["label"].value_counts()

print(label_counts)

label_counts.to_csv(
    OUTPUT_DIR / "label_distribution.csv"
)

plt.figure(figsize=(8, 5))

label_counts.plot(kind="bar")

plt.title("Fake and Original Review Distribution")
plt.xlabel("Review Label")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "label_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 6. CATEGORY DISTRIBUTION
# ==========================================

print("\n========== CATEGORY DISTRIBUTION ==========")

category_counts = df["category"].value_counts()

print(category_counts)

category_counts.to_csv(
    OUTPUT_DIR / "category_distribution.csv"
)

plt.figure(figsize=(12, 6))

category_counts.plot(kind="bar")

plt.title("Product Category Distribution")
plt.xlabel("Product Category")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "category_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 7. RATING DISTRIBUTION
# ==========================================

print("\n========== RATING DISTRIBUTION ==========")

rating_counts = df["rating"].value_counts().sort_index()

print(rating_counts)

rating_counts.to_csv(
    OUTPUT_DIR / "rating_distribution.csv"
)

plt.figure(figsize=(8, 5))

rating_counts.plot(kind="bar")

plt.title("Review Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rating_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 8. REVIEW LENGTH ANALYSIS
# ==========================================

print("\n========== REVIEW LENGTH ANALYSIS ==========")

df["review_length"] = df["text_"].fillna("").astype(str).apply(len)

print(df["review_length"].describe())

df["review_length"].describe().to_csv(
    OUTPUT_DIR / "review_length_summary.csv"
)

plt.figure(figsize=(9, 5))

plt.hist(df["review_length"], bins=30)

plt.title("Review Length Distribution")
plt.xlabel("Review Length (Characters)")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "review_length_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 9. RATING ANALYSIS BY LABEL
# ==========================================

print("\n========== RATING ANALYSIS BY LABEL ==========")

label_rating = df.groupby("label")["rating"].agg(
    ["count", "mean", "median"]
)

print(label_rating)

label_rating.to_csv(
    OUTPUT_DIR / "label_rating_summary.csv"
)

# ==========================================
# 10. FINAL DATASET SUMMARY
# ==========================================

print("\n========== FINAL DATASET SUMMARY ==========")

print("Total Reviews:", len(df))

print("Total Categories:", df["category"].nunique())

print("Total Labels:", df["label"].nunique())

print("Total Ratings:", df["rating"].nunique())

print("\nAll graphs and CSV reports have been saved successfully!")

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nMEMBER 1 DATASET EXPLORATION COMPLETED!")
