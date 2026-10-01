# Heart Disease Prediction --- Reproduction and Duplicate-Aware Stacking

## Author

**Shruti Kuhar**\
Master of Data Science (Professional)\
Deakin University

## Project Overview

This project reproduces and critically evaluates the machine-learning
approach presented in:

**M. Bhagat, A. Sharma, and P. Agarwal, "An efficient stacking-based
ensemble technique for early heart attack prediction," Multimedia Tools
and Applications, vol. 84, pp. 36351--36375, 2025. DOI:
10.1007/s11042-024-19293-7.**

The reproduction evaluates multiple machine-learning classifiers and a
stacking ensemble for heart disease prediction. A subsequent
data-quality audit identified a large number of exact duplicate
observations in the publicly available 1,025-row dataset. The project
therefore also develops and evaluates a duplicate-aware modelling
pipeline.

## Problem Definition and Research Question

Early heart-disease prediction is formulated here as a **binary
classification problem**: given 13 clinical predictor variables, the
model predicts the binary `target` indicating the presence or absence
of heart disease.

The methodological problem investigated in this reproduction is equally
important. The publicly distributed dataset contains many exact repeated
observations. If duplicate records are randomly divided between training
and testing sets, a model may be evaluated on feature vectors that it
has effectively already encountered during training. This can inflate
hold-out performance and weaken conclusions about generalisation to
genuinely unseen observations.

Accordingly, this project has two objectives:

1. reproduce the selected stacking-based heart-disease study as closely
   as possible using the available methodological information; and
2. test whether the apparent performance remains credible after
   duplicate records are explicitly accounted for.

The central research question is:

> **How does a stacking ensemble perform when evaluated on unique,
> previously unseen observations, and can a duplicate-aware tuned
> stacking pipeline improve classification performance without
> overstating changes in ranking/discrimination ability?**

The project therefore separates **reproduction performance** from
**duplicate-aware generalisation performance** rather than treating a
high random-split accuracy as sufficient evidence of generalisation.

## Dataset

The Heart Disease dataset used in this project is accessed **directly
from its publicly available online source during execution of the
notebook**.

Dataset source:

https://raw.githubusercontent.com/GehadGad/Heart-disease-dataset/main/heart.csv

The dataset is loaded directly into a Pandas DataFrame using:

``` python
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/GehadGad/Heart-disease-dataset/main/heart.csv"

df = pd.read_csv(DATA_URL)
```

Therefore, the analysis does **not require a manually downloaded local
dataset file**. Running the notebook retrieves the data directly from
the source above.

### Dataset Information

The following checks are performed after loading the dataset:

``` python
print("Shape:", df.shape)
print("Duplicates:", df.duplicated().sum())
print("Unique rows:", len(df.drop_duplicates()))
```

For the dataset used in this experiment, these checks produce:

-   **1,025 observations**
-   **14 columns**
-   **13 predictor variables**
-   **1 binary target variable (`target`)**
-   **723 exact duplicate rows**
-   **302 unique rows**

The predictors include age, sex, chest pain type, resting blood
pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart
rate, exercise-induced angina, ST depression, slope, number of major
vessels, and thalassemia.

### Important Reproducibility Note

An active internet connection is required when the dataset-loading cell
is executed because the data is retrieved directly from the public
GitHub source.

The initial reproduction experiment uses the complete 1,025-row dataset.
During the subsequent data-quality investigation, exact duplicate
observations are identified programmatically.

For the proposed duplicate-aware experiment, duplicate rows are removed
using:

``` python
df_unique = df.drop_duplicates().reset_index(drop=True)
```

This produces **302 unique observations**.

No observations are manually removed or altered. The duplicate-removal
process is implemented programmatically within the notebook so that it
can be reproduced directly from the original online dataset.

## Models Evaluated

The reproduction evaluates:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   XGBoost
-   Gaussian Naive Bayes
-   K-Nearest Neighbours
-   Stacking Ensemble

The proposed duplicate-aware stacking framework combines:

-   Logistic Regression
-   Tuned Random Forest
-   Tuned XGBoost
-   Logistic Regression as the meta-classifier

## Experimental Workflow

### Part 1 --- Reproduction

1.  Retrieve the 1,025-row dataset directly from the online source.
2.  Inspect the data and target distribution.
3.  Create an 80:20 stratified train-test split.
4.  Train the six individual classifier families.
5.  Construct a stacking ensemble.
6.  Evaluate Accuracy, Precision, Recall, F1, ROC-AUC, Balanced
    Accuracy, and MCC.
7.  Compare the reproduced results with the published results.

### Data-Quality and Train-Test Overlap Investigation

The dataset is then examined for exact duplicate observations and
train-test feature-vector overlap.

The analysis identifies:

-   723 exact duplicate rows.
-   302 unique rows.
-   202 of the 205 observations in the initial test set have feature
    vectors represented in the training data.

This diagnostic is used to investigate whether random row-level
partitioning can produce overly optimistic performance on this public
dataset representation.

This finding should not be interpreted as proof that the original
authors used the same split or experienced the same overlap, because
their exact train-test indices are unavailable.

### Part 2 --- Proposed Duplicate-Aware Method

1.  Remove exact duplicate observations before partitioning.
2.  Create a new stratified train-test split from the 302 unique
    observations.
3.  Verify zero exact feature-vector overlap between the new training
    and test sets.
4.  Tune Random Forest and XGBoost using cross-validation.
5.  Combine Logistic Regression, tuned Random Forest, and tuned XGBoost
    in a stacking ensemble.
6.  Use Logistic Regression as the final meta-classifier.
7.  Evaluate the proposed method on the duplicate-free hold-out set.
8.  Perform repeated stratified validation to examine model stability.

## Key Results

### Initial Reproduction

The initial random-split reproduction produced very high performance for
several flexible models, including:

-   Decision Tree accuracy: approximately **98.54%**
-   Random Forest accuracy: **100%**
-   XGBoost accuracy: **100%**
-   Stacking Ensemble accuracy: **100%**

These results motivated the duplicate and train-test overlap
investigation.

### Duplicate-Aware Hold-Out Evaluation

On the unique-data split:

-   Original-style stacking accuracy: approximately **75.41%**
-   Proposed tuned stacking accuracy: approximately **80.33%**
-   Proposed model recall: approximately **84.85%**
-   Proposed model F1: approximately **82.35%**
-   Proposed model ROC-AUC: approximately **86.58%**

The proposed model improves hold-out accuracy by approximately **4.92
percentage points** relative to the original-style stacking baseline
evaluated on the same unique-data split.

### Repeated Validation

Across the repeated stratified validation experiment, the proposed model
achieved approximately:

-   Accuracy: **83.44%**
-   Recall: **90.25%**
-   F1-score: **85.53%**
-   ROC-AUC: **90.42%**

The strongest improvement was observed in recall and other
threshold-dependent classification metrics. In the comparison highlighted
in the technical report, ROC-AUC remains **0.8658**, despite improvements
in several threshold-dependent metrics. Therefore, these changes should
**not** be interpreted as evidence that the model improved its overall
ranking or discrimination ability. ROC-AUC evaluates how well predicted
scores rank positive observations above negative observations across
thresholds, whereas accuracy, precision, recall and F1 depend on the
chosen classification threshold. The appropriate conclusion is therefore
that classification decisions at the selected threshold improved, while
ranking/discrimination performance was unchanged in that comparison.

## Installation

### Recommended Environment

The notebook can be executed in **Google Colab**, which is the
recommended option.

Alternatively, use:

-   Python 3.10 or later
-   Jupyter Notebook / JupyterLab

Install the required packages with:

``` bash
pip install -r requirements.txt
```

## How to Run the Project

1.  Open `11_1HD_Machine_Learning_Task.ipynb` in Google Colab or Jupyter
    Notebook.
2.  Ensure that the environment has internet access.
3.  Install the packages listed in `requirements.txt` if they are not
    already available.
4.  Run the notebook cells sequentially from top to bottom.

The notebook will automatically:

-   retrieve the Heart Disease dataset from the documented public
    source;
-   inspect dataset dimensions and class distribution;
-   identify exact duplicate observations;
-   reproduce the individual machine-learning models;
-   construct the initial stacking ensemble;
-   calculate evaluation metrics;
-   investigate train-test feature-vector overlap;
-   create the unique-observation dataset;
-   tune Random Forest and XGBoost;
-   construct the proposed stacking ensemble;
-   evaluate the duplicate-aware hold-out experiment;
-   perform repeated validation; and
-   generate associated tables and figures.

**No separate manual dataset-download step is required to execute the
notebook.**

## Reproduction Assumptions

Some details required for exact computational reproduction, including
the original train-test indices, random seeds, and certain model
settings, were not fully specified in the publication.

Where required, reproducible implementation choices were therefore
adopted and documented in the notebook and technical report. These
assumptions should not be interpreted as the exact unpublished settings
used by the original authors.

## Important Interpretation

The purpose of the duplicate analysis is not to claim that the original
paper necessarily experienced the same train-test contamination. The
original train-test indices are not available.

Instead, the reproduction demonstrates that the publicly distributed
1,025-row dataset representation is highly vulnerable to train-test
overlap when random row-level partitioning is performed without first
accounting for exact duplicates.

The central methodological conclusion of this project is that **credible
evaluation and generalisation are more important than headline accuracy
alone**.

## Main Notebook

`11_1HD_Machine_Learning_Task.ipynb`

## Reference

M. Bhagat, A. Sharma, and P. Agarwal,\
"An efficient stacking-based ensemble technique for early heart attack
prediction,"\
*Multimedia Tools and Applications*, vol. 84, pp. 36351--36375, 2025.\
DOI: 10.1007/s11042-024-19293-7
