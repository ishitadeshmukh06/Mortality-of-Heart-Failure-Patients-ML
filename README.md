# 🫀 Clinical Prediction of Heart Failure Mortality

## 📖 Overview
This project develops a **predictive model** to identify high-risk heart failure patients based on clinical features. It aims to assist in early medical treatment and reduce mortality rates by comparing two machine learning approaches — **Support Vector Machine (SVM)** and **Artificial Neural Networks (ANN)**.

---

## 📁 Project Structure
```
MORTALITY OF HEART FAILURE PATIENTS/
│
├── Project1_Mortality of Heart Failure Patients.ipynb   # Main Jupyter Notebook
├── heart.csv                                             # Dataset
├── .gitignore
└── README.md
```

---

## 🗂️ Dataset
- **Source:** `heart.csv` (local file)
- **Size:** 918 patients
- **Target Variable:** `HeartDisease` (0 = Healthy, 1 = Heart Disease)
- **Features include:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis
- Class distribution of HeartDisease target variable
- Age distribution by disease status
- Swarm + box plots for all numerical features
- Count plots for all categorical features
- Correlation heatmap after one-hot encoding

### 2. Data Preprocessing
- Hidden missing values in `RestingBP` and `Cholesterol` (zeros) imputed with median
- One-hot encoding for all categorical features
- Feature scaling using `StandardScaler`
- 70/30 train-test split

### 3. Model Training & Evaluation
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) |
|---|---|---|---|
| Support Vector Machine (SVM) | 86% | 90% | 86% |
| Artificial Neural Network (ANN) | 86% | 91% | 85% |

> ✅ Both models perform similarly overall. **ANN** edges ahead on Precision while **SVM** has a slightly better Recall. For medical diagnostics where missing actual cases is costly, SVM's higher recall makes it slightly preferable.

### 4. ANN Architecture
- 3 Dense layers (16 → 8 → 8 units) with ReLU activation
- Dropout layers (0.25 and 0.5) to prevent overfitting
- Output layer with Sigmoid activation for binary classification
- EarlyStopping (patience=20) — training stopped at epoch 67

---

## 🛠️ Requirements

- Python 3.8+
- Jupyter Notebook / JupyterLab

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key libraries:
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `scikit-learn` — SVM, scaling, train-test split, evaluation metrics
- `tensorflow` / `keras` — ANN model building

---

## 🚀 Getting Started

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/mortality-heart-failure.git
   cd mortality-heart-failure
```
2. **Install dependencies**
```bash
   pip install -r requirements.txt
```
3. **Launch Jupyter**
```bash
   jupyter notebook
```
4. **Open and run** `Project1_Mortality of Heart Failure Patients.ipynb`

---

## 📊 Results

- Both SVM and ANN achieved **86% accuracy** on the test set
- **ANN** gave higher precision (91%) — fewer false alarms
- **SVM** gave higher recall (86%) — catches more actual heart disease cases
- Key predictors identified: `ST_Slope_Flat`, `ChestPainType_ASY`, `ExerciseAngina`, `MaxHR`

---

## 🚀 Future Improvements
- Try advanced models like Random Forest or XGBoost
- Add feature engineering and selection
- Tune ANN hyperparameters for better recall
- Address class imbalance with SMOTE or class weights

