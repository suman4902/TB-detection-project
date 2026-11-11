
# ============================================================
# 04_train_text_model.py
# Clinical Text-Based Classification
# Tuberculosis Project
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import seaborn as sns


# ------------------------------------------------------------
# 2. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(r"D:\TB_Project")

CSV_PATH = PROJECT_DIR / "data" / "dataset.csv"

MODEL_PATH = PROJECT_DIR / "text_model.pkl"


# ------------------------------------------------------------
# 3. LOAD THE DATASET
# ------------------------------------------------------------

# Load dataset containing image names, clinical text and labels

df = pd.read_csv(CSV_PATH)

# Exclude the corrupted image to maintain the same dataset
# used in the image-only experiment

df = df[
    ~df["image"].str.endswith("CHNCXR_0338_1.png")
].copy()

# Remove rows where clinical text is missing

df = df.dropna(subset=["clinical_text", "label"])

# Convert clinical text into string format

df["clinical_text"] = df["clinical_text"].astype(str)

print("Total usable text records:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())


# ------------------------------------------------------------
# 4. SEPARATE INPUT TEXT AND TARGET LABEL
# ------------------------------------------------------------

# Input variable:
# Clinical text associated with each chest X-ray

X = df["clinical_text"]

# Target variable:
# 0 = Normal
# 1 = Abnormal

y = df["label"].astype(int)


# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAINING, VALIDATION AND TEST SETS
# ------------------------------------------------------------

# Use the same splitting procedure as the image-only model.
# This allows a fair comparison between image and text models.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))


# ------------------------------------------------------------
# 6. CONVERT CLINICAL TEXT INTO NUMERICAL FEATURES
# ------------------------------------------------------------

# Machine learning algorithms cannot directly understand text.
#
# TF-IDF converts words into numerical values based on:
# 1. How frequently a word occurs in a document.
# 2. How important the word is across all documents.
#
# Unigrams = individual words
# Bigrams = pairs of consecutive words

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

# Learn vocabulary and calculate TF-IDF values
# using ONLY the training text.

X_train_tfidf = tfidf.fit_transform(X_train)

# Convert validation and test text using the same vocabulary.
# Do not fit the vectorizer again on these datasets.

X_val_tfidf = tfidf.transform(X_val)

X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF feature shapes:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Testing:", X_test_tfidf.shape)


# ------------------------------------------------------------
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

# Logistic Regression learns the relationship between
# clinical text features and the target class.

text_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train the model using training data

text_model.fit(
    X_train_tfidf,
    y_train
)

print("\nText model training completed.")


# ------------------------------------------------------------
# 8. EVALUATE MODEL ON VALIDATION DATA
# ------------------------------------------------------------

# Predict validation class probabilities

val_probabilities = text_model.predict_proba(
    X_val_tfidf
)[:, 1]

# Convert probabilities into class predictions
# using a threshold of 0.5

val_predictions = (
    val_probabilities >= 0.5
).astype(int)

# Calculate validation metrics

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0
)

print("\nVALIDATION PERFORMANCE")
print("--------------------------------")

print("Accuracy:", round(val_accuracy, 4))
print("Precision:", round(val_precision, 4))
print("Recall:", round(val_recall, 4))
print("F1-score:", round(val_f1, 4))


# ------------------------------------------------------------
# 9. FINAL EVALUATION ON TEST DATA
# ------------------------------------------------------------

# Predict probability of abnormal class

test_probabilities = text_model.predict_proba(
    X_test_tfidf
)[:, 1]

# Convert probabilities into class labels

test_predictions = (
    test_probabilities >= 0.5
).astype(int)

# Calculate performance metrics

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

print("\nTEXT MODEL TEST PERFORMANCE")
print("--------------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))


# ------------------------------------------------------------
# 10. GENERATE CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Abnormal"],
    yticklabels=["Normal", "Abnormal"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title("Confusion Matrix - Text Model")

plt.tight_layout()

cm_path = PROJECT_DIR / "text_confusion_matrix.png"

plt.savefig(cm_path, dpi=150)

plt.close()


# ------------------------------------------------------------
# 11. GENERATE ROC CURVE
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_probabilities
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")

plt.title("ROC Curve - Clinical Text Model")

plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = PROJECT_DIR / "text_roc_curve.png"

plt.savefig(roc_path, dpi=150)

plt.close()


# ------------------------------------------------------------
# 12. GENERATE PRECISION-RECALL CURVE
# ------------------------------------------------------------

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_test,
    test_probabilities
)

avg_precision = average_precision_score(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    pr_recall,
    pr_precision,
    label=f"Average Precision = {avg_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve - Clinical Text Model")

plt.legend()
plt.grid(True)
plt.tight_layout()

pr_path = PROJECT_DIR / "text_precision_recall_curve.png"

plt.savefig(pr_path, dpi=150)

plt.close()


# ------------------------------------------------------------
# 13. DETAILED CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("--------------------------------")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Normal", "Abnormal"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 14. SAVE THE TRAINED TEXT MODEL
# ------------------------------------------------------------

# Save both the TF-IDF vectorizer and Logistic Regression model.
# Both are required when making predictions on new text.

import joblib

joblib.dump(
    {
        "vectorizer": tfidf,
        "model": text_model
    },
    MODEL_PATH
)

print("\nText model saved to:", MODEL_PATH)


# ------------------------------------------------------------
# 15. DISPLAY GENERATED OUTPUT FILES
# ------------------------------------------------------------

print("\nTEXT MODEL EVALUATION COMPLETED")

print("Confusion matrix:", cm_path)

print("ROC curve:", roc_path)

print("Precision-Recall curve:", pr_path)

print(df["label"].value_counts())


# ------------------------------------------------------------
# CHECK 1: INSPECT CLINICAL TEXT SAMPLES
# ------------------------------------------------------------

print("\nSAMPLE CLINICAL TEXT RECORDS")
print("--------------------------------")

# Display five normal and five abnormal clinical reports

for label in [0, 1]:

    print(f"\nLABEL: {label}")

    samples = df[df["label"] == label]["clinical_text"].head(5)

    for i, text in enumerate(samples, start=1):

        print(f"\nSample {i}:")
        print(text)


# ------------------------------------------------------------
# CHECK 2: CHECK WHETHER TEXT IS DUPLICATED
# ------------------------------------------------------------

print("\nDUPLICATE TEXT CHECK")
print("--------------------------------")

print(
    "Number of unique clinical texts:",
    df["clinical_text"].nunique()
)

print(
    "Number of duplicate clinical texts:",
    df["clinical_text"].duplicated().sum()
)


# ------------------------------------------------------------
# CHECK 3: CHECK WORD FREQUENCIES BY CLASS
# ------------------------------------------------------------

print("\nCOMMON WORDS IN EACH CLASS")
print("--------------------------------")

from sklearn.feature_extraction.text import CountVectorizer

for label in [0, 1]:

    class_text = df[df["label"] == label]["clinical_text"]

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=None,
        max_features=20
    )

    word_matrix = vectorizer.fit_transform(class_text)

    word_counts = word_matrix.sum(axis=0).A1

    words = vectorizer.get_feature_names_out()

    word_frequency = pd.DataFrame({
        "Word": words,
        "Frequency": word_counts
    })

    word_frequency = word_frequency.sort_values(
        by="Frequency",
        ascending=False
    )

    print(f"\nLabel {label}:")

    print(word_frequency.to_string(index=False))


# ------------------------------------------------------------
# CHECK 4: INVESTIGATE DUPLICATE CLINICAL TEXT
# ------------------------------------------------------------

# Find all records having duplicate clinical text
duplicates = df[
    df.duplicated(subset=["clinical_text"], keep=False)
].copy()

print("\nDUPLICATE TEXT INVESTIGATION")
print("--------------------------------")

print("Total records with repeated text:", len(duplicates))

# Check whether identical text has different labels
label_counts = duplicates.groupby("clinical_text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1]

print(
    "Number of identical texts with conflicting labels:",
    len(conflicting_texts)
)

# Display a few duplicate examples with their labels
print("\nSample duplicate records:")

print(
    duplicates[
        ["case_id", "clinical_text", "label"]
    ].sort_values("clinical_text").head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 16. GENERATE TEXT PROBABILITIES FOR THE SAME TEST CASES
# ------------------------------------------------------------

import joblib

# Load image predictions to identify the exact 100 test cases
image_results = pd.read_csv(
    PROJECT_DIR / "image_test_predictions.csv"
)

# Recreate the test records used by the text model
text_test_df = df.loc[X_test.index].copy()

# Verify that the text model used the same test cases
if set(text_test_df["case_id"]) != set(image_results["case_id"]):
    raise ValueError(
        "Image and text test sets do not match. "
        "Check the dataset splits before combining predictions."
    )

# Load the saved TF-IDF vectorizer and text model
saved_text_model = joblib.load(MODEL_PATH)

vectorizer = saved_text_model["vectorizer"]
model = saved_text_model["model"]

# Convert the test clinical text into TF-IDF features
test_text_tfidf = vectorizer.transform(
    text_test_df["clinical_text"]
)

# Predict probability of abnormality
text_probabilities = model.predict_proba(
    test_text_tfidf
)[:, 1]

# Prepare the text prediction results
text_results = pd.DataFrame({
    "case_id": text_test_df["case_id"].values,
    "actual_label": text_test_df["label"].values,
    "text_probability": text_probabilities
})

# Save text probabilities
text_results_path = PROJECT_DIR / "text_test_predictions.csv"

text_results.to_csv(
    text_results_path,
    index=False
)

print("\nTEXT PROBABILITIES GENERATED")
print("--------------------------------")

print("Number of test records:", len(text_results))

print("Text predictions saved to:", text_results_path)

print("\nFirst five text predictions:")
print(text_results.head())
