
import re
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. LOAD DATASET
# ==================================================

DATA_PATH = r"D:\TB_Project\data\dataset.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["clinical_text", "label"]).copy()
df = df.reset_index(drop=True)

print("Dataset shape:", df.shape)


# ==================================================
# 2. REMOVE DIAGNOSIS-REVEALING WORDS
# ==================================================

diagnosis_terms = [
    "normal",
    "tuberculosis",
    "ptb",
    "stb",
    "natb",
    "atb",
    "tb"
]

pattern = r"\b(" + "|".join(diagnosis_terms) + r")\b"


def clean_clinical_text(text):
    text = str(text).lower()
    text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["revised_text"] = df["clinical_text"].apply(clean_clinical_text)

# Use the actual model input as the grouping variable.
# Identical revised texts must remain in the same split.
groups = df["revised_text"]

X = df["revised_text"]
y = df["label"]


# ==================================================
# 3. GROUP-BASED TRAIN / VALIDATION / TEST SPLIT
# Approximate proportions: 70% / 15% / 15%
# ==================================================

# First separate 15% for testing
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

# Then separate 15% of the complete dataset for validation
# from the remaining 85%.
train_val_df = df.iloc[train_val_idx]

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15 / 0.85,
    random_state=42
)

train_relative_idx, val_relative_idx = next(
    split2.split(
        train_val_df["revised_text"],
        train_val_df["label"],
        groups=train_val_df["revised_text"]
    )
)

train_idx = train_val_idx[train_relative_idx]
val_idx = train_val_idx[val_relative_idx]

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()

X_train = train_df["revised_text"]
y_train = train_df["label"]

X_val = val_df["revised_text"]
y_val = val_df["label"]

X_test = test_df["revised_text"]
y_test = test_df["label"]

print("\n========== GROUP-BASED SPLIT ==========")

print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Testing records:", len(test_df))

print("\nClass distribution:")
print("Training:\n", y_train.value_counts().sort_index())
print("Validation:\n", y_val.value_counts().sort_index())
print("Testing:\n", y_test.value_counts().sort_index())


# ==================================================
# 4. VERIFY NO IDENTICAL TEXT OVERLAP
# ==================================================

train_texts = set(X_train)
val_texts = set(X_val)
test_texts = set(X_test)

print("\n========== DUPLICATE OVERLAP CHECK ==========")

print("Training-validation overlap:",
      len(train_texts.intersection(val_texts)))

print("Training-testing overlap:",
      len(train_texts.intersection(test_texts)))

print("Validation-testing overlap:",
      len(val_texts.intersection(test_texts)))


# ==================================================
# 5. TF-IDF TEXT CONVERSION
# ==================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)


# ==================================================
# 6. TRAIN LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed.")


# ==================================================
# 7. VALIDATION RESULTS
# ==================================================

val_predictions = model.predict(X_val_tfidf)

print("\n========== VALIDATION RESULTS ==========")

print("Accuracy:",
      accuracy_score(y_val, val_predictions))

print("F1-score:",
      f1_score(y_val, val_predictions, zero_division=0))


# ==================================================
# 8. TEST RESULTS
# ==================================================

test_predictions = model.predict(X_test_tfidf)
test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

print("\n========== GROUP-BASED TEXT TEST RESULTS ==========")

print("Accuracy:",
      accuracy_score(y_test, test_predictions))

print("Precision:",
      precision_score(y_test, test_predictions, zero_division=0))

print("Recall:",
      recall_score(y_test, test_predictions, zero_division=0))

print("F1-score:",
      f1_score(y_test, test_predictions, zero_division=0))

if y_test.nunique() == 2:
    print("ROC-AUC:",
          roc_auc_score(y_test, test_probabilities))
else:
    print("ROC-AUC: Cannot calculate because test set has only one class.")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))


# ==================================================
# 9. SAVE MODEL
# ==================================================

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model
    },
    r"D:\TB_Project\group_based_text_model.pkl"
)


# ==================================================
# 10. SAVE TEST PREDICTIONS
# ==================================================

test_results = pd.DataFrame({
    "case_id": test_df["case_id"].values,
    "actual_label": y_test.values,
    "group_text_probability": test_probabilities
})

test_results.to_csv(
    r"D:\TB_Project\group_based_text_predictions.csv",
    index=False
)

print("\nSaved group_based_text_model.pkl")
print("Saved group_based_text_predictions.csv")

print("\n========== EXPERIMENT COMPLETED ==========")
