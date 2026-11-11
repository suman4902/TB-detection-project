
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image


# ============================================================
# 2. DEFINE DATASET FOLDER PATHS
# ============================================================

# Main data folder
base = Path(r"D:\TB_Project\data")

# Folder containing chest X-ray images
image_dir = base / "CXR_png"

# Folder containing clinical reading text files
text_dir = base / "ClinicalReadings"


# ============================================================
# 3. COUNT AND LIST IMAGE FILES
# ============================================================

# Find all PNG images in the image folder
files = sorted(image_dir.glob("*.png"))

# Count the number of images and text files
print("Images:", len(files))
print("Text files:", len(list(text_dir.glob("*.txt"))))

# Display the first image file
print(files[0])

# Extract the filename without its extension
first_image_file = files[0]
print(first_image_file.name)

# Extract the case ID from the filename
# Example: CHNCXR_0001_0
case_id = first_image_file.stem
print(case_id)


# ============================================================
# 4. MATCH IMAGE WITH ITS CLINICAL TEXT FILE
# ============================================================

# Construct the corresponding text file path
# Example: CHNCXR_0001_0.txt
text_file = text_dir / (case_id + ".txt")
print(text_file)

# Read the clinical information from the text file
clinical_text = text_file.read_text(encoding="utf-8")
print(clinical_text)

# Extract the label from the last character of the case ID
# 0 = Normal, 1 = Abnormal
label = int(case_id[-1])
print(label)

# Display all information belonging to this case
print("ID:", case_id)
print("Image:", first_image_file)
print("Clinical text:", clinical_text)
print("Label:", label)


# ============================================================
# 5. CREATE A DATAFRAME FOR ALL CASES
# ============================================================

# Create an empty list to store information about each case
rows = []

# Go through every X-ray image
for image_file in files:

    # Extract the case ID from the image filename
    case_id = image_file.stem

    # Find the matching clinical text file
    text_file = text_dir / (case_id + ".txt")

    # Read the clinical text
    clinical_text = text_file.read_text(encoding="utf-8")

    # Extract the label from the filename
    label = int(case_id[-1])

    # Store all information about this case as a dictionary
    rows.append({
        "case_id": case_id,
        "image": str(image_file),
        "clinical_text": clinical_text,
        "label": label
    })

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(rows)

# Display the first five records
print(df.head())

# Display the total number of cases
print("Number of cases:", len(df))

# Count the number of normal and abnormal cases
print(df["label"].value_counts())

# Save the DataFrame as a CSV file
df.to_csv(base / "dataset.csv", index=False)

print("Dataset saved at:", base / "dataset.csv")
print("Dataset shape:", df.shape)


# ============================================================
# 6. EXPLORE AN EXAMPLE CHEST X-RAY IMAGE
# ============================================================

# Open the first X-ray image
img = Image.open(first_image_file)

# Display image dimensions (width, height)
print("Image size:", img.size)

# Display image mode (RGB, grayscale, palette, etc.)
print("Image mode:", img.mode)

# Open the image using the default image viewer
img.show()


# Display the first and last filenames
print("First image:", files[0].name)
print("Last image:", files[-1].name)


# ============================================================
# 7. FIND AND EXPLORE AN ABNORMAL IMAGE
# ============================================================

# Find the first image whose filename ends in label 1
tb_file = next(
    file for file in files if int(file.stem[-1]) == 1
)

print("Abnormal image:", tb_file.name)

# Open the abnormal image and convert it into RGB
tb_img = Image.open(tb_file).convert("RGB")

# Display its dimensions and mode
print("Size:", tb_img.size)
print("Mode:", tb_img.mode)

# Display the abnormal image
tb_img.show()

# Display the image using Matplotlib
plt.imshow(tb_img)
plt.axis("off")
plt.title("Abnormal-labelled Chest X-ray")
plt.show()


# ============================================================
# 8. CONVERT THE IMAGE INTO A NUMERICAL ARRAY
# ============================================================

# Convert the image into a NumPy array
image_array = np.array(tb_img)

# Display the array dimensions
print("Array shape:", image_array.shape)

# Display the data type of the pixel values
print("Data type:", image_array.dtype)

# Display a small section of the array
print(image_array[:5, :5])

# Display the pixel values at the top-left corner
print("Top-left pixel:", image_array[0, 0])


# ============================================================
# 9. NORMALIZE THE IMAGE PIXEL VALUES
# ============================================================

# Divide pixel values by 255 to convert them to a 0-1 range
normalized_array = image_array / 255.0

# Display the normalized top-left pixel
print("Normalized top-left pixel:", normalized_array[0, 0])


# ============================================================
# 10. RESIZE THE IMAGE
# ============================================================

# Resize the image to 224 x 224 pixels
resized_img = tb_img.resize((224, 224))

print("Resized image size:", resized_img.size)

# Convert the resized image into a NumPy array
resized_array = np.array(resized_img)

print("Resized array shape:", resized_array.shape)


# ============================================================
# 11. NORMALIZE THE RESIZED IMAGE
# ============================================================

# Normalize pixel values of the resized image
normalized_resized_array = resized_array / 255.0

print(
    "Normalized resized top-left pixel:",
    normalized_resized_array[0, 0]
)
