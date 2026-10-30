
# Product Taxonomy Intelligence 🏷️

A comprehensive **Natural Language Processing (NLP) and Machine Learning project for automated product category classification** using Python, Scikit-learn, and TensorFlow/Keras.

The project focuses on analyzing product titles, extracting meaningful textual features, developing and comparing classification models, evaluating prediction performance, and explaining model decisions through character-level feature analysis.

---

## 🌟 Features

### Core Features

* **Data Understanding & Cleaning:** Inspection of dataset structure, missing values, duplicates, product clusters, and merchant information.
* **Exploratory Data Analysis:** Analysis of category distributions, product title characteristics, merchant patterns, and category-specific terminology.
* **Data Preparation:** Text preparation, target encoding, and group-aware train-test splitting.
* **Feature Engineering:** Word-level, character-level, and combined TF-IDF representations.
* **Machine Learning Models:** Development and comparison of multiple classification algorithms.
* **Hyperparameter Optimization:** Model tuning using group-aware cross-validation.
* **Deep Learning:** TextCNN experimentation for product category classification.
* **Model Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and category-level performance.
* **Error Analysis:** Investigation of misclassifications, product title lengths, and decision margins.
* **Model Explainability:** Global coefficient importance, category-specific features, and local prediction explanations.
* **Business Insights:** Identification of opportunities for automated product categorization and catalog consistency.

---

## 🗂 Project Structure

The project is organized as follows:

```text
Product-Taxonomy-Intelligence/
│
├── data/
│   ├── raw/                                  # Original PriceRunner dataset
│   └── processed/                            # Prepared datasets and features
│       └── features/                         # TF-IDF matrices and encoded targets
│
├── notebooks/
│   ├── 01_data_understanding.ipynb           # Dataset inspection
│   ├── 02_exploratory_data_analysis.ipynb    # Exploratory Data Analysis
│   ├── 03_data_preparation.ipynb             # Data cleaning and splitting
│   ├── 04_feature_engineering.ipynb          # Text feature extraction
│   ├── 05_model_development.ipynb            # Classical ML experimentation
│   ├── 06_model_evaluation.ipynb             # Final model evaluation
│   ├── 07_model_explainability.ipynb         # Model interpretation
│   └── 08_textcnn_development.ipynb          # TextCNN experimentation
│
├── models/                                   # Fitted preprocessing artifacts
│
├── reports/
│   └── figures/                              # Project visualizations
│
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── .gitignore                                # Git ignored files
```

**Note:** The project retains fitted text vectorizers and label encoders for reproducibility. Trained classification models are not stored as serialized model files.

---

## 🏗️ Project Architecture

```text
+----------------------------------+
|            DATA LAYER            |
|                                  |
|  PriceRunner Product Dataset     |
|  Product Titles and Categories   |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|        DATA UNDERSTANDING        |
|                                  |
|  - Dataset inspection            |
|  - Missing values                |
|  - Duplicate analysis            |
|  - Product clusters              |
|  - Merchant information          |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|            EDA LAYER             |
|                                  |
|  - Category distributions        |
|  - Product title analysis        |
|  - Merchant patterns             |
|  - Cluster characteristics       |
|  - Category-specific keywords    |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|       DATA PREPARATION           |
|                                  |
|  - Data cleaning                 |
|  - Text preparation              |
|  - Group-aware train-test split  |
|  - Target encoding               |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|       FEATURE ENGINEERING        |
|                                  |
|  - Word-level TF-IDF             |
|  - Character-level TF-IDF        |
|  - Combined TF-IDF               |
|  - Sparse feature matrices       |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|        MODEL DEVELOPMENT         |
|                                  |
|  - Classical ML classifiers      |
|  - Group-aware cross-validation  |
|  - Hyperparameter optimization   |
|  - Model comparison              |
|  - TextCNN experimentation       |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|        MODEL EVALUATION          |
|                                  |
|  - Accuracy and F1-score         |
|  - Category-level performance    |
|  - Confusion matrix              |
|  - Misclassification analysis    |
|  - Decision margin analysis      |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|       MODEL EXPLAINABILITY       |
|                                  |
|  - Character coefficients        |
|  - Global feature importance     |
|  - Local feature contributions   |
|  - Competing category scores     |
|  - Explanation validation        |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|         BUSINESS INSIGHTS        |
|                                  |
|  - Automated categorization      |
|  - Catalog consistency           |
|  - Ambiguous product detection   |
|  - Model limitations             |
+----------------------------------+
```

---

## 🛠 Tech Stack

* **Language:** Python 3.11
* **Environment:** Jupyter Notebook, VS Code
* **Libraries:**
  * **Data Manipulation:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn
  * **Machine Learning:** Scikit-learn
  * **Text Feature Extraction:** TF-IDF Vectorizer
  * **Deep Learning:** TensorFlow/Keras
  * **Sparse Matrix Processing:** SciPy
  * **Experiment Tracking:** MLflow
  * **Artifact Persistence:** Joblib
* **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

The project uses the **Product Classification and Clustering dataset** from the UCI Machine Learning Repository, originally sourced from PriceRunner.

The dataset contains **35,311 product listings** across **10 product categories**.

### Dataset Features

```text
Product ID
Product Title
Merchant ID
Cluster ID
Cluster Label
Category ID
Category Label
```

### Target Variable

**Category Label**

The objective is to predict the appropriate product category using the textual information contained in `Product Title`.

### Product Categories

1. CPUs
2. Digital Cameras
3. Dishwashers
4. Freezers
5. Fridge Freezers
6. Fridges
7. Microwaves
8. Mobile Phones
9. TVs
10. Washing Machines

### Dataset Summary

| Component | Value |
|---|---:|
| Original observations | 35,311 |
| Observations after preparation | 35,310 |
| Original features | 7 |
| Product categories | 10 |
| Unique product titles | 30,993 |
| Unique merchants | 306 |
| Unique product clusters | 13,233 |
| Training observations | 28,919 |
| Testing observations | 6,391 |

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering)

---

## 🎯 Problem Statement

E-commerce catalogs contain large numbers of product listings with inconsistent naming conventions, abbreviations, technical specifications, and product identifiers.

Manual classification can become inefficient as catalogs grow, while overlapping terminology between related products can introduce categorization errors.

The objective of this project is to develop a **multiclass NLP classification system** that accurately categorizes products using their titles while providing interpretable explanations for model predictions.

---

## 🚀 Project Objectives

* Understand product category distributions and textual characteristics.
* Identify meaningful patterns in product titles.
* Investigate product clusters and repeated listings.
* Prepare text data for supervised classification.
* Develop word-level and character-level TF-IDF representations.
* Train and compare multiple machine learning classifiers.
* Optimize model performance through cross-validation.
* Evaluate TextCNN as a deep learning alternative.
* Assess performance using classification metrics and error analysis.
* Explain model decisions using character-level feature contributions.
* Identify practical opportunities for product catalog automation.

---

## 🔍 Exploratory Data Analysis

The EDA stage investigates the structure, composition, and textual characteristics of the product dataset.

### Analysis Areas

* Product category distribution
* Product title character lengths
* Product title word counts
* Numerical and special character usage
* Duplicate product titles
* Product cluster distributions
* Merchant representation
* Merchant-category relationships
* Category-specific keywords
* Product naming patterns

### Key Observations

* **Fridge Freezers** represent the largest category, accounting for approximately **15.58%** of product listings.
* Product titles contain approximately **53 characters** and **8.67 words** on average.
* Approximately **95%** of titles contain numerical characters, highlighting the importance of technical specifications and model identifiers.
* The dataset contains **13,233 product clusters**, including numerous products represented by multiple listings.
* Related product categories share terminology, creating potential classification ambiguity.

These findings guided text preparation, feature engineering, and the group-aware validation strategy.

---

## ⚙️ Data Preparation & Feature Engineering

### Data Preparation

The preparation workflow included:

* Dataset cleaning and validation.
* Selection of product titles as predictive inputs.
* Identification of related product listings.
* Group-aware train-test splitting.
* Target label encoding.
* Preparation of reusable processed datasets.

### Text Feature Engineering

Three feature representations were explored.

#### Word-Level TF-IDF

Captures informative words and word combinations associated with product categories.

#### Character-Level TF-IDF

Captures recurring character sequences, abbreviations, technical specifications, and model identifiers.

#### Combined TF-IDF

Combines word-level and character-level representations to evaluate complementary textual information.

### Final Feature Representation

The selected configuration uses:

| Component | Configuration |
|---|---|
| Feature representation | Character TF-IDF |
| Feature dimensions | 50,000 |
| Target encoding | Label Encoding |
| Training observations | 28,919 |
| Testing observations | 6,391 |

Character-level TF-IDF proved particularly effective for capturing product naming patterns and technical identifiers.

---

## 🤖 Machine Learning Approach

The project treats product categorization as a **supervised multiclass classification problem**.

Multiple classical machine learning models were evaluated using different text feature representations.

### Modeling Workflow

```text
Product Title Data
        |
        v
Data Preparation
        |
        v
Group-Aware Train-Test Split
        |
        v
Text Feature Engineering
        |
        v
Multiple Classification Models
        |
        v
5-Fold Group-Aware Cross-Validation
        |
        v
Hyperparameter Optimization
        |
        v
Model Comparison
        |
        v
Final Ridge Classifier
        |
        v
Independent Test Evaluation
```

### Cross-Validation Strategy

The project uses **5-fold StratifiedGroupKFold cross-validation** to maintain category representation while separating related product groups.

**Macro F1-score** was used as a primary model comparison metric.

### Selected Model

**Ridge Classifier + Character TF-IDF**

| Parameter | Value |
|---|---|
| Model | Ridge Classifier |
| Regularization alpha | 0.5 |
| Feature representation | Character TF-IDF |
| Cross-validation Macro F1 | 0.9879 |
| Cross-validation standard deviation | 0.0021 |

The selected model demonstrated strong classification performance and consistent validation results.

---

## 🧠 Deep Learning Experimentation

A **Text Convolutional Neural Network (TextCNN)** was evaluated as a deep learning alternative.

The experiment investigated whether convolutional text representations could improve product category classification.

### Cross-Validation Comparison

| Model | Macro F1-score |
|---|---:|
| Ridge Classifier + Character TF-IDF | **0.9879** |
| TextCNN | 0.9333 |

### Key Findings

* TextCNN successfully learned useful patterns from product titles.
* The Ridge Classifier achieved stronger cross-validation performance.
* Classical machine learning provided an effective and computationally efficient solution for this dataset.

---

## 📈 Model Evaluation

The selected Ridge Classifier was evaluated on an independent test dataset containing **6,391 observations**.

### Final Model Performance

| Metric | Result |
|---|---:|
| Accuracy | **98.81%** |
| Macro Precision | 98.88% |
| Macro Recall | 98.84% |
| Macro F1-score | **98.86%** |
| Weighted F1-score | 98.81% |

### Prediction Summary

| Component | Result |
|---|---:|
| Total test observations | 6,391 |
| Correct predictions | 6,315 |
| Incorrect predictions | 76 |
| Error rate | 1.19% |

### Evaluation Analysis

The evaluation included:

* Overall classification metrics.
* Category-level precision, recall, and F1-score.
* Confusion matrix analysis.
* Misclassification investigation.
* Product title length analysis.
* Decision margin analysis.

### Key Evaluation Findings

* The classifier achieved strong performance across all 10 categories.
* **Fridges** recorded the lowest category F1-score at **0.9658**.
* The most notable confusion involved **Fridges** and **Fridge Freezers**.
* Product titles containing 1–20 characters had the highest observed error rate at **3.16%**.
* Correct predictions generally exhibited larger decision margins than incorrect predictions.

---

## 🔎 Model Explainability

Model explainability was performed using the Ridge Classifier's learned coefficients and character-level TF-IDF feature contributions.

### Explainability Methods

* Category-specific positive and negative coefficients.
* Global character feature importance.
* Local feature contribution analysis.
* Comparison of competing category decision scores.
* Mathematical validation of local explanations.

### Representative Character Patterns

| Category | Influential patterns |
|---|---|
| CPUs | `intel`, `ghz`, `i7` |
| Dishwashers | `dish`, `sms` |
| Fridge Freezers | `kgn`, `idge` |
| Microwaves | `wave`, `hmt` |
| Mobile Phones | `hone`, `gb` |
| TVs | `tv`, `kd` |
| Washing Machines | `wash`, `kg` |

### Local Prediction Analysis

An Android device containing processor-related terminology was incorrectly classified as a CPU.

The prediction was influenced by character patterns such as:

* `ghz`
* `core`
* `hz`

The competing category scores differed by just **0.0028**, indicating an ambiguous classification.

### Explainability Validation

Local feature contributions were mathematically validated by reconstructing the Ridge Classifier's decision scores from feature contributions and intercepts.

The reconstructed scores matched the original model outputs.

---

## 💡 Key Findings

* Product titles contain substantial category information through technical terminology, brand names, and model identifiers.
* Character-level TF-IDF proved highly effective for product classification.
* The selected Ridge Classifier achieved **98.81% test accuracy** and **98.86% Macro F1-score**.
* Classical machine learning outperformed the evaluated TextCNN configuration.
* Closely related refrigeration categories presented the greatest classification challenges.
* Short product titles generally exhibited higher error rates.
* Decision margins helped distinguish clearer predictions from ambiguous cases.
* Character-level coefficient analysis provided interpretable explanations of model behavior.

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

* The dataset covers only 10 predefined product categories.
* Product titles may contain ambiguous or incomplete information.
* Closely related categories share terminology and technical specifications.
* Character-level TF-IDF does not explicitly capture broader semantic meaning.
* Generalization to new merchants, unseen products, and changing catalog conventions remains unverified.
* Ridge Classifier decision scores are not calibrated probabilities.

### Future Improvements

* Expand the dataset with more diverse product categories.
* Incorporate additional product descriptions and metadata where appropriate.
* Explore pretrained language models and semantic text embeddings.
* Improve classification of closely related product categories.
* Evaluate calibrated confidence estimates and human-review thresholds.
* Assess robustness on unseen merchants and newer product listings.
* Develop a production classification pipeline with model monitoring and drift detection.

---

## 📁 Notebook Workflow

The project is organized into eight notebooks.

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Dataset inspection and initial understanding |
| `02_exploratory_data_analysis.ipynb` | Exploratory Data Analysis |
| `03_data_preparation.ipynb` | Data cleaning and group-aware splitting |
| `04_feature_engineering.ipynb` | TF-IDF feature extraction |
| `05_model_development.ipynb` | Classical ML development and tuning |
| `06_model_evaluation.ipynb` | Final model evaluation and error analysis |
| `07_model_explainability.ipynb` | Global and local model interpretation |
| `08_textcnn_development.ipynb` | TextCNN experimentation |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/TejasMJ/Product-Taxonomy-Intelligence.git
cd Product-Taxonomy-Intelligence
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and follow the numerical workflow.

**Note:** The TextCNN notebook is a separate experimentation workflow. Classical model evaluation and explainability use the selected Ridge Classifier.

The original dataset and required processed artifacts must be available in their corresponding project directories.

---

## 📌 Project Status

**Status:** Completed ✅

The project includes:

* Data understanding and cleaning.
* Exploratory Data Analysis.
* Group-aware data preparation.
* NLP feature engineering.
* Classical machine learning experimentation.
* Hyperparameter optimization.
* TextCNN experimentation.
* Final model evaluation.
* Classification error analysis.
* Model explainability.

**Final Selected Model:** Ridge Classifier with Character TF-IDF.

**Test Accuracy:** 98.81%.

**Macro F1-score:** 98.86%.

---

## 👨‍💻 Author

**Tejas Jadhav**

* GitHub: [@tejas-jadhav](https://github.com/TejasMJ)
* LinkedIn: [Tejas Jadhav](https://www.linkedin.com/in/tejas-m-jadhav/)
