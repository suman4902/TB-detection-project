
import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==================================================
# 1. SETTINGS
# ==================================================

SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001

PROJECT_DIR = r"D:\TB_Project"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "dataset.csv")
IMAGE_DIR = os.path.join(PROJECT_DIR, "data", "CXR_png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)


# ==================================================
# 2. LOAD DATA AND VERIFY IMAGES
# ==================================================

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["clinical_text", "label", "image"]).copy()
df = df.reset_index(drop=True)

def get_image_path(filename):
    filename = str(filename)

    # First try the filename inside CXR_png
    path1 = os.path.join(IMAGE_DIR, os.path.basename(filename))

    # Also support a path relative to the project folder
    path2 = os.path.join(PROJECT_DIR, filename)

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None


df["image_path"] = df["image"].apply(get_image_path)

# Remove missing or unreadable images before splitting
valid_indices = []

for idx, row in df.iterrows():
    path = row["image_path"]

    if path is None:
        continue

    try:
        with Image.open(path) as img:
            img.verify()

        valid_indices.append(idx)

    except Exception:
        print("Skipping unreadable image:", row["image"])

df = df.loc[valid_indices].copy()
df = df.reset_index(drop=True)

df["label"] = df["label"].astype(int)

print("\nUsable dataset:", len(df))
print("Class distribution:")
print(df["label"].value_counts().sort_index())


# ==================================================
# 3. REMOVE DIRECT DIAGNOSIS TERMS FROM TEXT
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


def clean_text(text):
    text = str(text).lower()
    text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["revised_text"] = df["clinical_text"].apply(clean_text)


# ==================================================
# 4. CREATE COMMON STRATIFIED GROUP SPLIT
# ==================================================

X = df["revised_text"]
y = df["label"]
groups = df["revised_text"]

outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

train_val_idx, test_idx = next(
    outer_splitter.split(X, y, groups)
)

train_val_df = df.iloc[train_val_idx].copy()
test_df = df.iloc[test_idx].copy()

inner_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
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

print("\n========== COMMON DATA SPLIT ==========")

for name, part in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(f"\n{name}: {len(part)} records")
    print(part["label"].value_counts().sort_index())

# Confirm no duplicate revised text crosses splits
print("\n========== OVERLAP CHECK ==========")

print(
    "Train-Test:",
    len(set(train_df["revised_text"]) &
        set(test_df["revised_text"]))
)

print(
    "Train-Validation:",
    len(set(train_df["revised_text"]) &
        set(val_df["revised_text"]))
)

print(
    "Validation-Test:",
    len(set(val_df["revised_text"]) &
        set(test_df["revised_text"]))
)


# ==================================================
# 5. TRAIN TEXT MODEL
# ==================================================

print("\n========== TEXT MODEL ==========")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_text = vectorizer.fit_transform(train_df["revised_text"])
X_val_text = vectorizer.transform(val_df["revised_text"])
X_test_text = vectorizer.transform(test_df["revised_text"])

text_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=SEED
)

text_model.fit(X_train_text, train_df["label"])

text_probabilities = text_model.predict_proba(X_test_text)[:, 1]

joblib.dump(
    {"vectorizer": vectorizer, "model": text_model},
    os.path.join(PROJECT_DIR, "final_text_model.pkl")
)

print("Text model training completed.")


# ==================================================
# 6. IMAGE DATASET AND TRANSFORMATIONS
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class CXRDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return image, label


train_dataset = CXRDataset(train_df, transform)
val_dataset = CXRDataset(val_df, transform)
test_dataset = CXRDataset(test_df, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 7. LOAD RESNET-18
# ==================================================

print("\n========== IMAGE MODEL ==========")

image_model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

# Freeze pretrained feature extraction layers
for parameter in image_model.parameters():
    parameter.requires_grad = False

# Replace final classification layer
image_model.fc = nn.Linear(
    image_model.fc.in_features,
    2
)

image_model = image_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    image_model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# 8. TRAIN IMAGE MODEL
# ==================================================

for epoch in range(EPOCHS):

    image_model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = image_model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    # Validation loss
    image_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = image_model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {average_loss:.4f} "
        f"| Validation Loss: {val_loss:.4f}"
    )


# ==================================================
# 9. IMAGE MODEL TEST PREDICTIONS
# ==================================================

image_model.eval()

image_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = image_model(images)

        probabilities = torch.softmax(outputs, dim=1)[:, 1]

        image_probabilities.extend(
            probabilities.cpu().numpy()
        )

image_probabilities = np.array(image_probabilities)

torch.save(
    image_model.state_dict(),
    os.path.join(PROJECT_DIR, "final_image_model.pth")
)

print("Image model training completed.")


# ==================================================
# 10. COMBINE IMAGE AND TEXT PROBABILITIES
# ==================================================

# Both prediction arrays correspond to test_df
combined_probabilities = (
    image_probabilities + text_probabilities
) / 2

actual_labels = test_df["label"].to_numpy()

image_predictions = (image_probabilities >= 0.5).astype(int)
text_predictions = (text_probabilities >= 0.5).astype(int)
combined_predictions = (
    combined_probabilities >= 0.5
).astype(int)


# ==================================================
# 11. EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, actual, predicted, probabilities):

    print(f"\n========== {name} ==========")

    print("Accuracy:", accuracy_score(actual, predicted))

    print(
        "Precision:",
        precision_score(actual, predicted, zero_division=0)
    )

    print(
        "Recall:",
        recall_score(actual, predicted, zero_division=0)
    )

    print(
        "F1-score:",
        f1_score(actual, predicted, zero_division=0)
    )

    if len(np.unique(actual)) == 2:
        print("ROC-AUC:", roc_auc_score(actual, probabilities))
    else:
        print("ROC-AUC: Not available (only one test class)")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))


# ==================================================
# 12. COMPARE ALL THREE MODELS
# ==================================================

evaluate_model(
    "IMAGE-ONLY MODEL",
    actual_labels,
    image_predictions,
    image_probabilities
)

evaluate_model(
    "TEXT-ONLY MODEL",
    actual_labels,
    text_predictions,
    text_probabilities
)

evaluate_model(
    "MULTIMODAL MODEL",
    actual_labels,
    combined_predictions,
    combined_probabilities
)


# ==================================================
# 13. SAVE CASE-BY-CASE PREDICTIONS
# ==================================================

results = pd.DataFrame({
    "case_id": test_df["case_id"].values,
    "actual_label": actual_labels,

    "image_probability": image_probabilities,
    "image_prediction": image_predictions,
    "image_match": image_predictions == actual_labels,

    "text_probability": text_probabilities,
    "text_prediction": text_predictions,
    "text_match": text_predictions == actual_labels,

    "combined_probability": combined_probabilities,
    "combined_prediction": combined_predictions,
    "combined_match": combined_predictions == actual_labels
})

results.to_csv(
    os.path.join(PROJECT_DIR, "final_multimodal_predictions.csv"),
    index=False
)


# ==================================================
# 14. SAVE FINAL COMPARISON TABLE
# ==================================================

metrics = []

for name, predicted, probabilities in [
    ("Image Only", image_predictions, image_probabilities),
    ("Text Only", text_predictions, text_probabilities),
    ("Image + Text", combined_predictions, combined_probabilities)
]:

    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(actual_labels, predicted),
        "Precision": precision_score(
            actual_labels, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual_labels, predicted, zero_division=0
        ),
        "F1_Score": f1_score(
            actual_labels, predicted, zero_division=0
        ),
        "ROC_AUC": (
            roc_auc_score(actual_labels, probabilities)
            if len(np.unique(actual_labels)) == 2
            else np.nan
        ),
        "Correct_Predictions": int(
            (predicted == actual_labels).sum()
        ),
        "Total_Test_Records": len(actual_labels)
    })

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    os.path.join(PROJECT_DIR, "final_model_comparison.csv"),
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(metrics_df.to_string(index=False))

print("\nCorrect prediction counts:")
print(results[
    ["image_match", "text_match", "combined_match"]
].sum())

print("\nSaved files:")
print("final_image_model.pth")
print("final_text_model.pkl")
print("final_multimodal_predictions.csv")
print("final_model_comparison.csv")

print("\n========== FINAL EXPERIMENT COMPLETED ==========")
