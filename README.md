# Pulsar Star Classification using Logistic Regression

## 📌 Project Overview
This project focuses on building a **binary classification model** to identify **pulsar stars** from astronomical signal data.  
The notebook demonstrates a complete **end-to-end machine learning workflow**, including EDA, preprocessing, model training, and evaluation.

The objective is to predict whether a given observation corresponds to a **pulsar star (`1`)** or **non-pulsar (`0`)**.

---

## 📂 Dataset
- **File:** `pulsar_data_train.csv`
- **Target column:** `target_class`
- **Features:** Statistical measures extracted from pulsar signal profiles and DM-SNR curves

---

## 🔍 Exploratory Data Analysis (EDA)
The following EDA steps were performed:

- Checked dataset shape, columns, and data types
- Verified:
  - Missing values
  - Duplicate rows
  - Unique values per column
- Statistical summary using `describe()`
- Correlation analysis using `df.corr()`

### Distribution & Skewness Analysis
- Feature distributions visualized using **Seaborn**
- Skewness calculated for numerical features
- Identified highly skewed features such as:
  - Excess kurtosis of integrated profile
  - Standard deviation of DM-SNR curve
  - Skewness of DM-SNR curve

---

## ⚖️ Class Imbalance Handling
- The target variable was found to be **imbalanced**
- Applied **undersampling** to balance both classes
- Dataset shuffled after balancing to avoid bias

---

## 🧹 Data Preprocessing

### Train–Test Split
- Features (`X`) and target (`y`) separated
- Data split into training and testing sets

### Missing Value Imputation
- Used **Median Imputation** with `SimpleImputer`
- `fit_transform` applied on training data
- `transform` applied on test data

### Feature Scaling
- Standardization applied using `StandardScaler`
- Scaled NumPy arrays converted back to Pandas DataFrames

---

## 🤖 Model Training

### Logistic Regression
- Algorithm: `LogisticRegression`
- Parameters:
  - `max_iter = 1000`
- Model trained on scaled and preprocessed data

---

## 📊 Model Evaluation
The model performance was evaluated using:

- **Accuracy Score**
- **Classification Report**
  - Precision
  - Recall
  - F1-score

These metrics help measure how effectively the model detects pulsar stars.

---

## 🛠️ Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## ✅ Key Takeaways
- EDA is essential to understand skewness and feature behavior
- Handling class imbalance improves model fairness
- Feature scaling is crucial for Logistic Regression
- A clean preprocessing pipeline leads to stable performance

---

## 🚀 Future Improvements
- Apply cross-validation
- Try advanced models (Random Forest, XGBoost)
- Use SMOTE instead of undersampling
- Perform feature selection

---

## 👤 Author
**Devendra Kushwah**  
B.Tech CSE (AI & ML)

---

⭐ If you found this project helpful, consider giving it a star!
