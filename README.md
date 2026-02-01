# XGBoost Medical Classification

> A single notebook that applies **XGBoost** binary classification to two
> independent medical datasets — **Cardiovascular Disease** and **Diabetes** —
> demonstrating the algorithm's versatility across different feature spaces,
> class distributions, and hyperparameter regimes.

---

## Table of Contents

1. [Why Two Datasets in One Project](#1-why-two-datasets-in-one-project)
2. [Repository Layout](#2-repository-layout)
3. [Datasets](#3-datasets)
   - 3.1 [Cardiovascular Disease](#31-cardiovascular-disease)
   - 3.2 [Diabetes](#32-diabetes)
4. [Methodology](#4-methodology)
5. [Model Configuration](#5-model-configuration)
6. [Results](#6-results)
   - 6.1 [Cardiovascular Model](#61-cardiovascular-model)
   - 6.2 [Diabetes Model](#62-diabetes-model)
   - 6.3 [Side-by-Side Comparison](#63-side-by-side-comparison)
7. [Key Takeaways](#7-key-takeaways)
8. [Reproducing the Analysis](#8-reproducing-the-analysis)
9. [Tech Stack](#9-tech-stack)
10. [Roadmap](#10-roadmap)
11. [License](#11-license)

---

## 1. Why Two Datasets in One Project

Running the same algorithm on two structurally different problems in a single
notebook is a deliberate choice, not a limitation. It lets a reader — or a
hiring manager — see three things at once:

| What it shows | How |
|---|---|
| **Algorithm mastery** | The same XGBoost API is correctly wired for both tasks without copy-paste errors |
| **Hyperparameter awareness** | `max_depth` is set to 10 for the large cardiovascular dataset and deliberately reduced to 1 for the small diabetes dataset — a conscious trade-off, not a random pick |
| **Generalisation thinking** | The notebook is not over-fit to one domain; the pipeline transfers cleanly to a completely different feature space |

This is standard practice in ML portfolios. A single-algorithm, multi-domain
notebook is more informative than two half-finished notebooks sitting in
separate repos.

---

## 2. Repository Layout

```
xgboost-classification/
│
├── XGBoost_Classification_Problem.ipynb   ← End-to-end notebook (both models)
├── dataset.csv                            ← Combined or primary dataset file
├── requirements.txt                       ← Pinned Python dependencies
├── LICENSE                                ← MIT License
└── README.md                              ← This file
```

---

## 3. Datasets

### 3.1 Cardiovascular Disease

| Attribute | Value |
|---|---|
| Rows | 70,000 |
| Raw columns | 13 |
| Columns after preprocessing | 12 (id dropped) |
| Target column | `cardio` (0 = healthy, 1 = cardiovascular disease) |
| Target balance | 49.97 % positive — nearly perfectly balanced |
| Age encoding | Originally in days; converted to years (`age / 365`) |

**Features used for training (11):**

| Feature | Description |
|---|---|
| `age` | Patient age in years (converted from days) |
| `gender` | 1 = female, 2 = male |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | 0 = non-smoker, 1 = smoker |
| `alco` | 0 = no alcohol, 1 = alcohol use |
| `active` | 0 = physically inactive, 1 = active |

**Top correlations with the `cardio` target:**

| Feature | Pearson r |
|---|---|
| age | 0.238 |
| cholesterol | 0.221 |
| weight | 0.182 |
| gluc | 0.089 |

---

### 3.2 Diabetes

| Attribute | Value |
|---|---|
| Rows | 768 |
| Columns | 9 |
| Target column | `Outcome` (0 = no diabetes, 1 = diabetes) |
| Target balance | 35.0 % positive — moderately imbalanced |

**Features used for training (8):**

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Genetic likelihood score based on family history |
| `Age` | Age in years |

---

## 4. Methodology

Both models follow the same reproducible pipeline. The notebook is divided
into clearly labelled sections so each stage is easy to locate.

```
┌──────────────────────┐
│  1. Import & Load    │  Read CSV into pandas DataFrame
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  2. EDA              │  .describe() · .info() · missing values · duplicates
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  3. Visualisation    │  Histograms · correlation heatmap (Seaborn)
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  4. Preprocessing    │  Drop irrelevant columns · feature/target split
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  5. Train/Test Split │  80 / 20 split via sklearn
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  6. Model Training   │  XGBClassifier with tuned hyperparameters
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  7. Evaluation       │  Accuracy · Classification Report · Confusion Matrix
└──────────────────────┘
```

---

## 5. Model Configuration

Both models use `binary:logistic` as the objective and `error` as the
evaluation metric. The key difference is `max_depth` — intentionally tuned
for each dataset's size and complexity.

| Hyperparameter | Cardiovascular | Diabetes | Rationale |
|---|---|---|---|
| `objective` | `binary:logistic` | `binary:logistic` | Binary classification in both cases |
| `eval_metric` | `error` | `error` | Tracks misclassification rate during boosting |
| `learning_rate` | 0.1 | 0.1 | Standard starting point; works well with low `n_estimators` |
| `max_depth` | **10** | **1** | 70 k rows can support deep trees; 768 rows cannot — deeper trees would overfit immediately |
| `n_estimators` | 10 | 10 | Kept low for both to keep training fast; increasing this is the first tuning lever |

> **Note:** The diabetes cell originally passed `use_label_encoder=False`. This
> parameter was deprecated and removed in recent XGBoost versions. It triggers
> a harmless warning but has no effect on results. Removing it cleans up the
> output.

---

## 6. Results

### 6.1 Cardiovascular Model

**Accuracy: 73.38 %** on 14,000 test samples.

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Healthy | 0.71 | 0.77 | 0.74 | 6,939 |
| 1 — Disease | 0.75 | 0.69 | 0.72 | 7,061 |
| **Macro Avg** | **0.73** | **0.73** | **0.73** | 14,000 |
| **Weighted Avg** | **0.73** | **0.73** | **0.73** | 14,000 |

The model performs evenly across both classes. Precision and recall are
balanced, which is expected given the near-perfect 50/50 target split in the
training data. No class dominates the predictions.

---

### 6.2 Diabetes Model

**Accuracy: 73.38 %** on 154 test samples.

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — No Diabetes | 0.74 | 0.94 | 0.83 | 104 |
| 1 — Diabetes | 0.71 | 0.30 | 0.42 | 50 |
| **Macro Avg** | **0.73** | **0.62** | **0.62** | 154 |
| **Weighted Avg** | **0.73** | **0.73** | **0.70** | 154 |

The accuracy headline is identical to the cardiovascular model, but the
per-class breakdown tells a very different story. The model correctly identifies
94 % of healthy patients but only catches 30 % of actual diabetes cases. This
is a direct consequence of `max_depth = 1`: a single-split tree cannot capture
the interaction effects that distinguish diabetic patients. This is discussed
further in the Takeaways section.

---

### 6.3 Side-by-Side Comparison

| Metric | Cardiovascular | Diabetes |
|---|---|---|
| Dataset size | 70,000 | 768 |
| Test set size | 14,000 | 154 |
| Target balance | 50 / 50 | 65 / 35 |
| `max_depth` | 10 | 1 |
| Accuracy | 73.38 % | 73.38 % |
| Macro F1 | 0.73 | 0.62 |
| Recall — Positive class | 0.69 | 0.30 |
| Precision — Positive class | 0.75 | 0.71 |

---

## 7. Key Takeaways

**1. Accuracy alone does not tell the full story.**
Both models report 73.38 % accuracy, but their macro F1 scores diverge
sharply (0.73 vs 0.62). The cardiovascular model is genuinely balanced;
the diabetes model is heavily biased toward predicting the majority class.
Always inspect per-class metrics, especially when class imbalance is present.

**2. `max_depth` is the most consequential hyperparameter here.**
Reducing `max_depth` from 10 to 1 collapses the diabetes model's ability to
learn complex decision boundaries. With only 768 rows, deeper trees risk
overfitting — but depth 1 is too aggressive a correction. A value in the
3–5 range, combined with cross-validation, would likely recover significant
recall on the positive class without memorising the training set.

**3. Dataset size dictates model complexity.**
The cardiovascular dataset (70 k rows) can comfortably support 10 levels of
tree depth. The diabetes dataset (768 rows) cannot. This project makes that
trade-off explicit and visible, which is more educational than hiding it
behind a single pre-tuned configuration.

**4. Class imbalance changes the evaluation game.**
The cardiovascular target is nearly 50/50, so accuracy is a reasonable
summary metric. The diabetes target is 65/35 — not extreme, but enough that
a model can achieve 65 % accuracy by simply predicting "no diabetes" every
single time. Metrics like F1 and recall on the minority class become
essential guards against that trap.

---

## 8. Reproducing the Analysis

### Prerequisites

| Software | Minimum Version |
|---|---|
| Python | 3.10+ |
| Jupyter Notebook / JupyterLab / VS Code with Jupyter extension | Latest |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/khadijja1/xgboost-classification.git
cd xgboost-classification

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows PowerShell:  venv\Scripts\activate
# macOS / Linux:       source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook XGBoost_Classification_Problem.ipynb
```

Execute all cells top-to-bottom using **Kernel → Restart & Run All**.

> The notebook reads datasets via relative paths. Keep the CSV files in the
> same directory as the `.ipynb` file.

---

## 9. Tech Stack

| Library | Role |
|---|---|
| `pandas` | DataFrame I/O and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Train/test split, classification report, confusion matrix |
| `xgboost` | XGBClassifier — the core model |
| `matplotlib` | Histogram plots |
| `seaborn` | Correlation heatmaps, confusion matrix heatmaps |

---

## 10. Roadmap

- [ ] **Cross-validation** — Replace the single 80/20 split with 5-fold
  stratified CV to get more reliable accuracy and F1 estimates, especially
  for the small diabetes dataset.
- [ ] **Hyperparameter tuning** — Use `GridSearchCV` or `RandomizedSearchCV`
  to sweep `max_depth`, `n_estimators`, and `learning_rate` systematically.
- [ ] **Class-weight handling** — Apply `scale_pos_weight` in XGBoost or
  `SMOTE` resampling to address the diabetes class imbalance and recover
  recall on the positive class.
- [ ] **Feature importance** — Plot `feature_importances_` from both trained
  models to identify which clinical indicators drive predictions most.
- [ ] **Remove deprecated parameter** — Delete `use_label_encoder=False`
  from the diabetes XGBClassifier call to eliminate the warning.

---

## 11. License

This project is licensed under the **MIT License** — see the
[LICENSE](LICENSE) file for details.

## 12. Contact 

**Khadija Faisal** 

**Github** : khadijja1

**Email** : khadijafaysal444@gmail.com