
# ============================================================
# 05_multimodal_evaluation.py
# Image + Clinical Text Prediction
# Tuberculosis Project
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    average_precision_score
)


# 1. DEFINE PROJECT PATHS

PROJECT_DIR = Path(r"D:\TB_Project")

IMAGE_PATH = PROJECT_DIR / "image_test_predictions.csv"
TEXT_PATH = PROJECT_DIR / "text_test_predictions.csv"


# 2. LOAD IMAGE AND TEXT PREDICTIONS

image_df = pd.read_csv(IMAGE_PATH)
text_df = pd.read_csv(TEXT_PATH)


# 3. MATCH IMAGE AND TEXT USING CASE ID

combined_df = pd.merge(
    image_df,
    text_df,
    on="case_id",
    suffixes=("_image", "_text")
)

# Check that all test records are matched
if len(combined_df) != 100:
    raise ValueError(
        f"Expected 100 matched records, found {len(combined_df)}"
    )

# Check that actual labels agree
if not (
    combined_df["actual_label_image"]
    == combined_df["actual_label_text"]
).all():
    raise ValueError("Actual labels do not match!")

combined_df["actual_label"] = combined_df["actual_label_image"]


print("Total matched test records:", len(combined_df))


# 4. CALCULATE COMBINED PROBABILITY

# Simple average of image and text probabilities

combined_df["combined_probability"] = (
    combined_df["image_probability"]
    + combined_df["text_probability"]
) / 2


# 5. CONVERT PROBABILITIES INTO CLASS PREDICTIONS

combined_df["combined_prediction"] = (
    combined_df["combined_probability"] >= 0.5
).astype(int)


# 6. CALCULATE PERFORMANCE METRICS

y_true = combined_df["actual_label"]

y_pred = combined_df["combined_prediction"]

y_prob = combined_df["combined_probability"]


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

roc_auc = auc(
    *roc_curve(y_true, y_prob)[:2]
)

avg_precision = average_precision_score(y_true, y_prob)


print("\nMULTIMODAL MODEL PERFORMANCE")
print("--------------------------------")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(avg_precision, 4))


# 7. GENERATE CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred)

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

plt.title("Confusion Matrix - Multimodal Model")

plt.tight_layout()

cm_path = PROJECT_DIR / "multimodal_confusion_matrix.png"

plt.savefig(cm_path, dpi=150)

plt.close()


# 8. GENERATE ROC CURVE

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

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
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Multimodal Model")

plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = PROJECT_DIR / "multimodal_roc_curve.png"

plt.savefig(roc_path, dpi=150)

plt.close()


# 9. DETAILED CLASSIFICATION REPORT

print("\nCLASSIFICATION REPORT")
print("--------------------------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Abnormal"],
        zero_division=0
    )
)


# 10. SAVE COMBINED PREDICTIONS

OUTPUT_PATH = PROJECT_DIR / "multimodal_test_predictions.csv"

combined_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nMULTIMODAL EVALUATION COMPLETED")

print("Combined predictions:", OUTPUT_PATH)
print("Confusion matrix:", cm_path)
print("ROC curve:", roc_path)


# ------------------------------------------------------------
# 11. COMPARE IMAGE-ONLY AND MULTIMODAL PREDICTIONS
# ------------------------------------------------------------

# Generate image-only predictions using a 0.5 threshold
combined_df["image_prediction"] = (
    combined_df["image_probability"] >= 0.5
).astype(int)

# Identify whether each model predicted correctly
combined_df["image_correct"] = (
    combined_df["image_prediction"]
    == combined_df["actual_label"]
)

combined_df["multimodal_correct"] = (
    combined_df["combined_prediction"]
    == combined_df["actual_label"]
)

# Identify cases where multimodal improved or worsened
improved_cases = combined_df[
    (~combined_df["image_correct"])
    & (combined_df["multimodal_correct"])
]

worsened_cases = combined_df[
    (combined_df["image_correct"])
    & (~combined_df["multimodal_correct"])
]

unchanged_cases = combined_df[
    combined_df["image_correct"]
    == combined_df["multimodal_correct"]
]

print("\nCASE-BY-CASE COMPARISON")
print("--------------------------------")

print("Cases improved by adding text:", len(improved_cases))
print("Cases worsened by adding text:", len(worsened_cases))
print("Cases with unchanged correctness:", len(unchanged_cases))

print("\nIMPROVED CASES:")
print(
    improved_cases[
        [
            "case_id",
            "actual_label",
            "image_probability",
            "text_probability",
            "combined_probability"
        ]
    ].to_string(index=False)
)

print("\nWORSENED CASES:")
print(
    worsened_cases[
        [
            "case_id",
            "actual_label",
            "image_probability",
            "text_probability",
            "combined_probability"
        ]
    ].to_string(index=False)
)

# Save detailed comparison
comparison_path = PROJECT_DIR / "model_case_comparison.csv"

combined_df.to_csv(comparison_path, index=False)

print("\nDetailed comparison saved to:", comparison_path)


# ------------------------------------------------------------
# 12. INSPECT CLINICAL TEXT OF IMPROVED CASES
# ------------------------------------------------------------

# Load original clinical reports
dataset_path = PROJECT_DIR / "data" / "dataset.csv"

original_df = pd.read_csv(dataset_path)

# Select case ID and clinical text
clinical_data = original_df[
    ["case_id", "clinical_text"]
]

# Match improved cases with their original clinical text
improved_text = improved_cases.merge(
    clinical_data,
    on="case_id",
    how="left"
)

print("\nCLINICAL TEXT OF IMPROVED CASES")
print("--------------------------------")

for _, row in improved_text.iterrows():

    print("\nCase ID:", row["case_id"])
    print("Actual label:", row["actual_label"])
    print("Image probability:", round(row["image_probability"], 4))
    print("Text probability:", round(row["text_probability"], 4))
    print("Clinical text:", row["clinical_text"])
    print("--------------------------------")
