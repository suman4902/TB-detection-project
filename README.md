# Multimodal Classification of Chest X-Ray Images and Clinical Text for Tuberculosis-Related Abnormality Detection

## 1. Project Overview

This project develops and evaluates a multimodal machine learning framework for classifying chest X-ray (CXR) images and associated clinical text into two dataset-defined categories:

- **Class 0:** Normal
- **Class 1:** Abnormal

The primary objective is to investigate whether combining visual information from chest X-ray images with clinical text can improve classification performance compared with using either modality independently.

The project uses the Shenzhen Chest X-ray dataset and explores three classification approaches:

1. Image-only classification using ResNet-18.
2. Text-only classification using TF-IDF and Logistic Regression.
3. Multimodal classification using equal-weight probability fusion.

## 2. Project Objectives

- Develop a CNN-based model for chest X-ray image classification.
- Develop a text classification model using clinical descriptions.
- Integrate image and text predictions through multimodal learning.
- Investigate potential information leakage caused by diagnosis-related terms and repeated clinical descriptions.
- Implement group-based data splitting to improve evaluation reliability.
- Compare the performance of image-only, text-only and multimodal models.

## 3. Project Architecture

The proposed framework consists of two independent processing branches followed by probability-level fusion.

### Image Processing Branch

- Chest X-ray image loading and preprocessing.
- Resizing to 224 × 224 pixels.
- Grayscale-to-three-channel conversion.
- Tensor conversion and ImageNet normalization.
- Feature extraction and classification using pretrained ResNet-18.

### Clinical Text Processing Branch

- Clinical text exploration and preprocessing.
- Removal of diagnosis-related terms to investigate potential target leakage.
- TF-IDF feature extraction using unigrams and bigrams.
- Binary classification using Logistic Regression.

### Multimodal Fusion

The predicted abnormal-class probabilities from both models are combined using equal-weight probability averaging.

**Fusion equation:**

P(combined) = 0.5 × P(image) + 0.5 × P(text)

A classification threshold of 0.5 is used to obtain the final predicted class.

## 4. Methodology

The project follows these major stages:

1. Dataset loading and exploratory data analysis.
2. Image availability and quality assessment.
3. Image preprocessing and ResNet-18 model development.
4. Clinical text exploration and preprocessing.
5. TF-IDF feature extraction and Logistic Regression.
6. Investigation of repeated text descriptions and conflicting labels.
7. Group-based data splitting using StratifiedGroupKFold.
8. Image and text probability generation.
9. Multimodal probability fusion.
10. Comparative evaluation using classification metrics.

## 5. Dataset Description

**Dataset:** Shenzhen Chest X-ray dataset.

The initial dataset contained 662 records. One unreadable image was identified during quality assessment and excluded.

The final dataset contained 661 usable records.

| Parameter | Details |
|---|---|
| Initial records | 662 |
| Final usable records | 661 |
| Class 0 (Normal) | 326 |
| Class 1 (Abnormal) | 335 |
| Image format | PNG |
| Image modality | Chest X-ray |
| Text modality | Clinical descriptions |

Each record contains an image reference, clinical text description and predefined binary class label.

The dataset itself is not included in this repository.

## 6. Data Leakage Investigation

During clinical text exploration, repeated descriptions and diagnosis-related terms were identified.

The investigation revealed:

- 456 unique original clinical text descriptions.
- 205 duplicate occurrences beyond the first occurrence.
- 17 conflicting revised text groups involving 102 records.

Diagnosis-related terms were removed to investigate the contribution of information beyond explicit label-related terminology.

Group-based splitting was subsequently implemented to prevent identical revised text descriptions from appearing across training, validation and test partitions.

The final common dataset split was:

| Dataset | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Training | 208 | 214 | 422 |
| Validation | 52 | 54 | 106 |
| Testing | 66 | 67 | 133 |
| **Total** | **326** | **335** | **661** |

No revised clinical text groups overlapped across these three partitions.

## 7. Experimental Results

All three final models were evaluated on the same test dataset containing 133 records.

### Comparative Model Performance

| Evaluation Metric | Image-only | Text-only | Multimodal |
|---|---:|---:|---:|
| Accuracy | 84.96% | 75.19% | **91.73%** |
| Precision | 84.06% | 68.89% | **88.89%** |
| Recall | 86.57% | 92.54% | **95.52%** |
| F1-score | 85.29% | 78.98% | **92.09%** |
| ROC-AUC | 92.70% | 93.74% | **96.61%** |

### Correct Predictions

| Model | Correct predictions | Incorrect predictions |
|---|---:|---:|
| Image-only | 113 | 20 |
| Text-only | 100 | 33 |
| Multimodal | 122 | 11 |

The multimodal model achieved the highest performance across all five evaluation metrics.

It correctly classified nine additional records compared with the image-only model and 22 additional records compared with the text-only model.

These results indicate the potential benefit of combining image and clinical text information for the selected classification task.

## 8. Technologies and Libraries

- Python
- PyTorch
- Torchvision
- ResNet-18
- Scikit-learn
- TF-IDF
- Logistic Regression
- Pandas
- NumPy
- PIL (Python Imaging Library)

## 9. Repository Contents

The repository contains the following implementation files:

| File | Description |
|---|---|
| 01_data_exploration.py | Dataset exploration |
| 02_train_image_model.py | Image model training |
| 03_evaluate_image_model.py | Image model evaluation |
| 04_train_text_model.py | Text model training |
| 05_multimodal_evaluation.py | Multimodal evaluation |
| 06_diagnosis_without_text.py | Diagnosis-term removal experiment |
| 07_group_based_text_experiment.py | Group-based text experiment |
| 08_stratified_group_text.py | Stratified group validation |
| 09_inspect_conflicting_text.py | Conflicting text investigation |
| 10_final_multimodal_experiment.py | Final multimodal experiment |
| Final_report.docx | Detailed project report |

## 10. Installation and Execution

### Prerequisites

- Python 3.x
- pip
- Required Python libraries
- Project dataset

### Dataset Organisation

Download and prepare the dataset separately. Organise it as follows:

```text
TB_Project/
│
├── data/
│   ├── dataset.csv
│   └── CXR_png/
│       ├── CHNCXR_0001_0.png
│       ├── CHNCXR_0002_0.png
│       └── ...
│
├── 01_data_exploration.py
├── 02_train_image_model.py
├── ...
└── 10_final_multimodal_experiment.py
```

### Execution Sequence

The Python scripts are numbered to indicate the progression of the experiments.

Start with dataset exploration and preprocessing, followed by image model development, text model development, leakage investigation, group-based validation and final multimodal evaluation.

Before execution, verify the dataset paths and configuration parameters in each script.

Install the required dependencies using the supplied requirements.txt file once available:

```bash
pip install -r requirements.txt
```

Note: The scripts may require configuration of local dataset paths and previously generated model artefacts.

## 11. Reproducibility

To reproduce the experiments:

1. Obtain the Shenzhen Chest X-ray dataset and corresponding clinical text data.
2. Organise the dataset according to the directory structure described above.
3. Install the required Python dependencies.
4. Execute the relevant preprocessing and model training scripts.
5. Generate image-only and text-only predictions.
6. Perform group-based data splitting using StratifiedGroupKFold.
7. Generate multimodal predictions through equal-weight probability averaging.
8. Evaluate all three models using the same test partition.
9. Compare the results using accuracy, precision, recall, F1-score and ROC-AUC.

The detailed implementation, experimental configurations and findings are documented in the final project report.

## 12. Limitations

- The final experiment uses a relatively small dataset of 661 usable records.
- Clinical text contains repeated descriptions and conflicting labels.
- The final performance is based on a single group-based test split.
- Equal-weight probability averaging is a simple fusion strategy.
- The image model uses a largely frozen pretrained ResNet-18 backbone.
- External validation is required to assess generalisation to other datasets.

## 13. Conclusion

The project demonstrates the implementation and comparative evaluation of image-only, text-only and multimodal classification approaches.

The multimodal model achieved an accuracy of 91.73%, an F1-score of 92.09% and a ROC-AUC of 96.61% on the final test dataset.

The findings suggest that image and clinical text predictions can provide complementary information for the classification of the dataset's predefined categories.

However, the results are preliminary and should not be interpreted as evidence of clinical diagnostic performance.

## 14. Academic Information

**Author:** Suman A L

**Programme:** M.Tech – Operations Research and Data Analytics (ORDA)

**Institute:** Indian Institute of Technology Kharagpur

**Department:** Industrial and Systems Engineering

**Course:** Programming for Data Science Laboratory

**Project Guide:** Prof. Subhajit Sidhanta

---

*This repository is intended for academic and research purposes. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical assessment.*
