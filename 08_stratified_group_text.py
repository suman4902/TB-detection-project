
import re
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
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

X = df["revised_text"]
y = df["label"]
groups = df["revised_text"]


# ==================================================
# 3. CHECK GROUPS WITH DIFFERENT LABELS
# ==================================================

label_counts = df.groupby("revised_text")["label"].nunique()

conflicting_groups = label_counts[label_counts > 1]

print("\nGroups with conflicting labels:", len(conflicting_groups))


# ==================================================
# 4. STRATIFIED GROUP TRAIN / VALIDATION / TEST SPLIT
# ==================================================

# First split: 80% train-validation, 20% test
outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    outer_splitter.split(X, y, groups)
)

train_val_df = df.iloc[train_val_idx].copy()
test_df = df.iloc[test_idx].copy()


# Second split: approximately 80% training and 20% validation
# within the remaining 80% of the dataset.
inner_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_relative_idx, val_relative_idx = next(
    inner_splitter.split(
        train_val_df["revised_text"],
        train_val_df["label"],
        train_val_df["revised_text"]
    )
)

train_df = train_val_df.iloc[train_relative_idx].copy()
val_df = train_val_df.iloc[val_relative_idx].copy()

X_train = train_df["revised_text"]
y_train = train_df["label"]

X_val = val_df["revised_text"]
y_val = val_df["label"]

X_test = test_df["revised_text"]
y_test = test_df["label"]


# ==================================================
# 5. DISPLAY SPLIT SIZES AND CLASS DISTRIBUTION
# ==================================================

print("\n========== STRATIFIED GROUP SPLIT ==========")

print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Testing records:", len(test_df))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_val.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ==================================================
# 6. VERIFY NO TEXT OVERLAP
# ==================================================

train_texts = set(X_train)
val_texts = set(X_val)
test_texts = set(X_test)

print("\n========== DUPLICATE OVERLAP CHECK ==========")

print(
    "Training-validation overlap:",
    len(train_texts.intersection(val_texts))
)

print(
    "Training-testing overlap:",
    len(train_texts.intersection(test_texts))
)

print(
    "Validation-testing overlap:",
    len(val_texts.intersection(test_texts))
)


# ==================================================
# 7. TF-IDF TEXT CONVERSION
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
# 8. TRAIN LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed.")


# ==================================================
# 9. VALIDATION RESULTS
# ==================================================

val_predictions = model.predict(X_val_tfidf)

print("\n========== VALIDATION RESULTS ==========")

print("Accuracy:", accuracy_score(y_val, val_predictions))

print(
    "F1-score:",
    f1_score(y_val, val_predictions, zero_division=0)
)


# ==================================================
# 10. TEST RESULTS
# ==================================================

test_predictions = model.predict(X_test_tfidf)

test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

print("\n========== STRATIFIED GROUP TEST RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, test_predictions))

print(
    "Precision:",
    precision_score(y_test, test_predictions, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, test_predictions, zero_division=0)
)

print(
    "F1-score:",
    f1_score(y_test, test_predictions, zero_division=0)
)

if y_test.nunique() == 2:
    print("ROC-AUC:", roc_auc_score(y_test, test_probabilities))
else:
    print("ROC-AUC cannot be calculated: test set contains one class.")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ==================================================
# 11. SAVE MODEL
# ==================================================

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model
    },
    r"D:\TB_Project\stratified_group_text_model.pkl"
)


# ==================================================
# 12. SAVE TEST PREDICTIONS
# ==================================================

test_results = pd.DataFrame({
    "case_id": test_df["case_id"].values,
    "actual_label": y_test.values,
    "stratified_group_probability": test_probabilities
})

test_results.to_csv(
    r"D:\TB_Project\stratified_group_text_predictions.csv",
    index=False
)

print("\nSaved stratified_group_text_model.pkl")
print("Saved stratified_group_text_predictions.csv")

print("\n========== EXPERIMENT COMPLETED ==========")
