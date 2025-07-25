# 🧠 Loan Acceptance Prediction Model

This project develops a machine learning model to predict the likelihood of customers accepting personal loan offers. It helps financial institutions, like Thera Bank, enhance targeted marketing strategies and improve offer acceptance rates.

## 📌 Objective

To identify potential customers who are more likely to accept a personal loan offer, using historical customer and account data. This allows the bank to minimize marketing costs and focus on high-potential leads.

## 📊 Dataset Overview

- **Source**: Thera Bank's customer data
- **Features**:
  - Age, Income, Experience, Family size
  - Education level, Credit card usage, Securities account
  - CD account, Online usage, Credit card use
- **Target**: `Personal Loan` (0 = No, 1 = Yes)

## ⚙️ Technologies Used

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data preprocessing
  - `seaborn`, `matplotlib` – Data visualization
  - `scikit-learn` – Machine Learning (KNN, Decision Trees, etc.)
  - `statsmodels` – Statistical analysis

## 🧪 Exploratory Data Analysis (EDA)

- Analyzed customer demographics and account behavior
- Visualized relationships between features and loan acceptance
- Identified key influencing factors (e.g., income, education)

## 🤖 Model Highlights

- Applied various classification algorithms
- **K-Nearest Neighbors (KNN)** gave best performance
- **Achieved 96% accuracy** on test data

## 📈 Results

- Built a predictive model with high accuracy
- Identified top features influencing loan acceptance
- Created a system that can be deployed for marketing decision support

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/loan-prediction-model.git
   cd loan-prediction-model
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:
Open loan_model.ipynb in Jupyter Notebook or VS Code

📎 File Structure
bash
Copy
Edit
loan-prediction-model/
│
├── loan_model.ipynb        # Main notebook with EDA + ML models
├── dataset.csv             # Input data file (optional for public upload)
├── requirements.txt        # Python packages used
└── README.md               # Project overview
