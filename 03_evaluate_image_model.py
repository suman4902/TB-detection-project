
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

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

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


# 1. Define project paths
PROJECT_DIR = Path(r"D:\TB_Project")
CSV_PATH = PROJECT_DIR / "data" / "dataset.csv"
MODEL_PATH = PROJECT_DIR / "resnet18_image_model.pth"


# 2. Load the same dataset and exclude the corrupted image
df = pd.read_csv(CSV_PATH)

df = df[
    ~df["image"].str.endswith("CHNCXR_0338_1.png")
].copy()


# 3. Recreate the same train, validation and test split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# 4. Prepare images exactly as during training
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. Create the dataset loader
class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.loc[index, "image"]
        label = int(self.dataframe.loc[index, "label"])
        case_id = self.dataframe.loc[index, "case_id"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return image, label, case_id


test_dataset = ChestXrayDataset(test_df, image_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# 6. Recreate ResNet-18 and load saved weights
device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model = model.to(device)
model.eval()


# 7. Predict probabilities for test images
true_labels = []
abnormal_probabilities = []
case_ids = []

with torch.no_grad():

    for images, labels, ids in test_loader:

        images = images.to(device)

        outputs = model(images)

        # Convert model outputs into probabilities
        probabilities = torch.softmax(outputs, dim=1)

        true_labels.extend(labels.numpy().tolist())

        abnormal_probabilities.extend(
            probabilities[:, 1].cpu().numpy().tolist()
        )

        case_ids.extend(ids)


print("Test images evaluated:", len(true_labels))


# 8. Save image probabilities for future multimodal integration
image_results = pd.DataFrame({
    "case_id": case_ids,
    "actual_label": true_labels,
    "image_probability": abnormal_probabilities
})

image_results_path = PROJECT_DIR / "image_test_predictions.csv"

image_results.to_csv(image_results_path, index=False)

print("Image predictions saved to:", image_results_path)
print("\nFirst five image predictions:")
print(image_results.head())


# 9. Generate the ROC curve
fpr, tpr, roc_thresholds = roc_curve(
    true_labels,
    abnormal_probabilities
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - ResNet-18 TB Image Model")
plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = PROJECT_DIR / "roc_curve.png"
plt.savefig(roc_path, dpi=150)
plt.close()


# 10. Generate the Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(
    true_labels,
    abnormal_probabilities
)

avg_precision = average_precision_score(
    true_labels,
    abnormal_probabilities
)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"Average Precision = {avg_precision:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - ResNet-18 TB Image Model")
plt.legend()
plt.grid(True)
plt.tight_layout()

pr_path = PROJECT_DIR / "precision_recall_curve.png"
plt.savefig(pr_path, dpi=150)
plt.close()


# 11. Display results
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print("ROC curve saved to:", roc_path)
print("Precision-Recall curve saved to:", pr_path)


# 12. Convert probabilities into predicted classes
predicted_labels = [
    1 if probability >= 0.5 else 0
    for probability in abnormal_probabilities
]


# 13. Calculate performance metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision_score_value = precision_score(true_labels, predicted_labels, zero_division=0)
recall_score_value = recall_score(true_labels, predicted_labels, zero_division=0)
f1 = f1_score(true_labels, predicted_labels, zero_division=0)

print("\nIMAGE MODEL PERFORMANCE")
print("----------------------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision_score_value, 4))
print("Recall:", round(recall_score_value, 4))
print("F1-score:", round(f1, 4))


# 14. Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

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
plt.title("Confusion Matrix - ResNet-18")

plt.tight_layout()

cm_path = PROJECT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()

print("Confusion matrix saved to:", cm_path)


# 15. Detailed classification report
print("\nClassification Report:")
print(classification_report(
    true_labels,
    predicted_labels,
    target_names=["Normal", "Abnormal"],
    zero_division=0
))
