# Swiss-Weather-Forecasting
Perfetto. Ti preparo un **README.md in stile GitHub**, professionale e pronto da inserire nel repository del progetto. Ho sintetizzato le parti chiave del tuo report (dataset, metodologia, modelli, risultati, ecc.).

---

# Machine Learning for Temperature Forecasting

## Overview

This project develops machine learning models to **forecast air temperature in Bern (Switzerland) 24 hours ahead** using meteorological data collected from multiple Swiss weather stations.

Due to Switzerland’s complex terrain and strong microclimatic variability, temperature forecasting presents significant challenges for traditional statistical methods. This project explores a **data-driven machine learning approach**, comparing linear models, tree-based ensembles, and gradient boosting techniques.

The project was developed for the **Machine Learning course at the University of Geneva (GSEM)**.

**Authors**

* Giorgio Bonetta
* Dawid Dymarczyk
* Radmehr Karimian

Course: Machine Learning – University of Geneva
Lecturer: Sebastian Engelke
Semester: Fall 2025 

---

# Problem Statement

The goal is to **predict the air temperature in Bern 24 hours into the future** using meteorological observations collected across Switzerland.

The challenge includes:

* **Spatial generalization**: Bern is not included among training stations.
* **Temporal structure**: hourly meteorological dynamics.
* **Data quality issues**: noise, missing values, and outliers.
* **Complex nonlinear relationships** among atmospheric variables.

The problem is formulated as a **supervised learning regression task**. 

---

# Dataset

The dataset comes from the Kaggle competition:

**S-403011 ML UNIGE Weather Prediction**

### Training data

* **7,579 hourly observations**
* **92 meteorological features**
* Measurements from **10 Swiss weather stations**

Stations include:

* Basel
* Genève
* Interlaken
* Sion
* St. Gallen
* Davos
* Zermatt
* Andermatt
* La Dôle
* Lugano

### Target variables

Temperature in Bern at multiple horizons:

* +12 hours
* +24 hours (**main target**)
* +48 hours

### Example predictors

* Wind speed and gust
* Solar radiation
* Sunshine duration
* Atmospheric pressure
* Precipitation
* Temperature
* Relative humidity
* Station identifier
* Timestamp

The **test dataset contains 3,247 observations** with no target values and is used for out-of-sample predictions. 

---

# Methodology

The modeling pipeline follows three main stages:

## 1. Data Preprocessing

Key steps:

* Removal of duplicate observations
* Handling missing values

  * median imputation (numerical)
  * mode imputation (categorical)
* One-hot encoding of categorical variables
* Feature alignment between train and test sets
* Leakage prevention

## 2. Feature Engineering

Several transformations were applied to improve model performance:

### Distribution transformations

* `log(1 + x)` for highly skewed variables
* `sqrt(x)` for moderate skewness

### Zero-inflation decomposition

Variables with many zeros (e.g., precipitation) were decomposed into:

* binary occurrence indicator
* magnitude conditional on occurrence

### Spatial features

Temperature differences between stations:

[
\Delta T_{s,t} = T_{s,t} - T_{Bern,t}
]

These features capture **temperature gradients across locations**.

### Temporal signals

* lagged temperature
* seasonal indicators

---

# Evaluation Protocol

Model performance is evaluated using:

* **5-fold cross-validation**
* **Mean Absolute Error (MAE)**

MAE was chosen because:

* it is robust to outliers
* errors are expressed directly in **degrees Celsius**

---

# Models Implemented

## 1. Linear Models

Baseline models include:

* Ordinary Least Squares (OLS)
* Ridge Regression
* Lasso Regression
* Elastic Net
* Least Absolute Deviations (LAD)
* **Huber Regression**

Best linear result:

```
Huber Regression
MAE ≈ 1.97
```

---

## 2. Bagging-Based Tree Models

### Random Forest

### Extra Trees

Example configuration:

* up to **1000 trees**
* maximum depth up to **30**
* randomized feature selection

Best tuned Random Forest:

```
MAE ≈ 1.88
```

However, performance improvements were limited despite extensive hyperparameter tuning.

---

## 3. Gradient Boosting Models

Boosting models achieved the best performance.

Implemented algorithms:

* Histogram Gradient Boosting
* **LightGBM**
* XGBoost

Best cross-validated performance:

```
LightGBM
MAE ≈ 1.79
```

---

## 4. Model Blending

A blended ensemble combining:

* LightGBM
* Huber regression

was tested using **out-of-fold predictions**.

Optimal blending weight:

```
Huber weight ≈ 0.11
```

Resulting performance:

```
MAE ≈ 1.791
```

---

## 5. Uncertainty Estimation

To estimate prediction uncertainty, **quantile regression with LightGBM** was implemented:

Quantiles:

* 0.1
* 0.5
* 0.9

This produces prediction intervals:

```
[ŷ₀.₁(x), ŷ₀.₉(x)]
```

---

# Key Results

| Model                       | MAE       |
| --------------------------- | --------- |
| Median baseline             | 6.72      |
| Ridge Regression            | ~1.99     |
| Huber Regression            | ~1.97     |
| Random Forest (tuned)       | ~1.88     |
| Histogram Gradient Boosting | ~1.84     |
| **LightGBM**                | **~1.79** |

Gradient boosting methods significantly outperform linear models and bagging-based trees.

---

# Key Findings

1. **Linear models provide strong interpretable baselines** but cannot capture nonlinear atmospheric dynamics.

2. **Random Forest improves accuracy**, but shows diminishing returns with deeper trees.

3. **Gradient boosting models dominate performance**, especially LightGBM.

4. **Feature engineering benefits boosting models**, but has little impact on bagging-based trees.

5. Blending models with different inductive biases can yield **small additional improvements**.

---

# Technologies Used

* Python
* Scikit-learn
* LightGBM
* XGBoost
* NumPy
* Pandas
* Matplotlib

---

# Applications

Accurate short-term temperature forecasts are useful for:

* **Energy demand planning**
* **Transportation management**
* **Agriculture**
* **Weather risk management**
