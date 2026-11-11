
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
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

    # Remove direct diagnosis terms
    text = re.sub(pattern, " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["revised_text"] = df["clinical_text"].apply(clean_clinical_text)

print("\nExamples of revised text:")
print(
    df[["clinical_text", "revised_text"]]
    .head(10)
    .to_string(index=False)
)

# Check how many reports became empty
empty_count = (df["revised_text"].str.strip() == "").sum()

print("\nReports with empty revised text:", empty_count)


# ==================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# Same split settings as previous experiments
# ==================================================

X = df["revised_text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nDataset split:")
print("Training cases:", len(X_train))
print("Validation cases:", len(X_val))
print("Testing cases:", len(X_test))


# ==================================================
# 4. CONVERT TEXT INTO NUMBERS USING TF-IDF
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
# 5. TRAIN LOGISTIC REGRESSION MODEL
# ==================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nRevised text model training completed.")


# ==================================================
# 6. VALIDATION RESULTS
# ==================================================

val_predictions = model.predict(X_val_tfidf)

print("\n========== VALIDATION RESULTS ==========")

print(
    "Accuracy:",
    accuracy_score(y_val, val_predictions)
)

print(
    "F1-score:",
    f1_score(y_val, val_predictions, zero_division=0)
)


# ==================================================
# 7. TEST RESULTS
# ==================================================

test_predictions = model.predict(X_test_tfidf)

test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

print("\n========== REVISED TEXT MODEL: TEST RESULTS ==========")

print(
    "Accuracy:",
    accuracy_score(y_test, test_predictions)
)

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

print(
    "ROC-AUC:",
    roc_auc_score(y_test, test_probabilities)
)

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
# 8. SAVE REVISED MODEL
# ==================================================

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model
    },
    r"D:\TB_Project\revised_text_model.pkl"
)


# ==================================================
# 9. SAVE TEST PREDICTIONS
# ==================================================

test_results = pd.DataFrame({
    "case_id": df.loc[X_test.index, "case_id"].values,
    "actual_label": y_test.values,
    "revised_text_probability": test_probabilities
})

test_results.to_csv(
    r"D:\TB_Project\revised_text_test_predictions.csv",
    index=False
)

print("\nSaved revised_text_model.pkl")
print("Saved revised_text_test_predictions.csv")

print("\n========== EXPERIMENT COMPLETED ==========")


import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = r"D:\TB_Project\data\dataset.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["clinical_text", "label"]).copy()

# Clean text consistently
df["clean_text"] = (
    df["clinical_text"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("Total records:", len(df))


# ==========================================
# 2. RECREATE THE ORIGINAL SPLIT
# ==========================================

X = df["clean_text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)


# ==========================================
# 3. CHECK TRAIN-TEST TEXT OVERLAP
# ==========================================

train_texts = set(X_train)
test_texts = set(X_test)

overlapping_texts = train_texts.intersection(test_texts)

overlap_test_cases = X_test.isin(train_texts).sum()

print("\n========== TRAIN-TEST OVERLAP ==========")

print("Unique training texts:", len(train_texts))
print("Unique testing texts:", len(test_texts))

print("Identical texts in both sets:", len(overlapping_texts))

print(
    "Test records whose text also occurs in training:",
    overlap_test_cases
)

print(
    "Percentage of test records with repeated training text:",
    round(overlap_test_cases / len(X_test) * 100, 2),
    "%"
)


# ==========================================
# 4. DISPLAY OVERLAPPING EXAMPLES
# ==========================================

print("\n========== OVERLAPPING TEXT EXAMPLES ==========")

overlap_examples = df[
    df["clean_text"].isin(overlapping_texts)
][["case_id", "clean_text", "label"]]

print(overlap_examples.head(20).to_string(index=False))


# ==========================================
# 5. CHECK LABEL CONSISTENCY
# ==========================================

label_counts = (
    df.groupby("clean_text")["label"]
    .nunique()
)

conflicting_texts = label_counts[label_counts > 1]

print("\n========== LABEL CONSISTENCY ==========")

print(
    "Identical texts with conflicting labels:",
    len(conflicting_texts)
)

print("\nDuplicate-text check completed.")
