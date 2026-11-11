
import re
import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = r"D:\TB_Project\data\dataset.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["clinical_text", "label"]).copy()

# ==========================================
# 2. REMOVE DIAGNOSIS TERMS
# ==========================================

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

# ==========================================
# 3. IDENTIFY CONFLICTING TEXT GROUPS
# ==========================================

label_counts = df.groupby("revised_text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

conflict_df = df[
    df["revised_text"].isin(conflicting_texts)
].copy()

# ==========================================
# 4. DISPLAY SUMMARY
# ==========================================

print("Total conflicting text groups:", len(conflicting_texts))
print("Total records in conflicting groups:", len(conflict_df))

print("\nRecords per conflicting group:")
print(
    conflict_df.groupby("revised_text")
    .size()
    .sort_values(ascending=False)
)

# ==========================================
# 5. DISPLAY ORIGINAL AND REVISED TEXT
# ==========================================

print("\n========== CONFLICTING REPORT DETAILS ==========")

for i, (text, group) in enumerate(
    conflict_df.groupby("revised_text"), start=1
):
    print("\n----------------------------------------")
    print(f"GROUP {i}")
    print("Revised text:", repr(text))

    print("\nOriginal reports:")

    for _, row in group.iterrows():
        print(
            "Case ID:", row["case_id"],
            "| Label:", row["label"],
            "| Original text:", repr(row["clinical_text"])
        )

# ==========================================
# 6. SAVE RESULTS
# ==========================================

conflict_df.to_csv(
    r"D:\TB_Project\conflicting_text_groups.csv",
    index=False
)

print("\nSaved conflicting_text_groups.csv")
print("\nInspection completed.")
