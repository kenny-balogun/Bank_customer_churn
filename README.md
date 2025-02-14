# Predicting Customer Churn: Machine Learning Solutions for Banking Retention  

## Overview  

Customer churn poses a significant challenge for banks, leading to revenue loss and increased customer acquisition costs.  
This project applies **machine learning models** to predict customer churn, leveraging insights to improve **customer retention strategies**.  

The models were implemented using **Python (Scikit-Learn) and Azure Machine Learning (Azure ML)**, allowing for a **comprehensive comparison** between traditional coding-based approaches and cloud-based no-code/low-code solutions.  

## Objectives  

- Develop predictive models to **identify customers at risk of churning**.  
- Compare the effectiveness of **K-Nearest Neighbors (KNN), Random Forest, and Neural Networks**.  
- Implement and evaluate models using both **Python and Azure ML**.  
- Provide actionable **business recommendations** to enhance customer retention.  

## Dataset  

The dataset was obtained from **Lloyds Bank's Forage Program** and consists of **five spreadsheets** merged into a single dataset.  
It includes **6,812 observations** and **17 features**, categorized as:  

- **Demographic Information** (Age, Income Level, Marital Status, Gender)  
- **Transaction History** (Spending Patterns, Frequency of Transactions)  
- **Customer Service Interactions** (Complaints, Resolutions)  
- **Online Activity** (Login Frequency, Mobile Banking Usage)  
- **Churn Status** (Indicating whether a customer has left the bank)  

### Ethical Considerations  

- **Data Anonymization:** Personally identifiable information was removed.  
- **Fairness & Compliance:** The study aligns with **GDPR regulations** to ensure ethical data usage.  

## Exploratory Data Analysis (EDA)  

Key findings from the data exploration phase:  

- **Churn Rate:** 20% of customers have churned, creating a **class imbalance**.  
- **Feature Importance:**  
  - Customers with **low login frequency** were more likely to churn.  
  - Older customers and those with **lower spending habits** exhibited higher churn rates.  
- **Correlation Analysis:** Weak correlation between numerical features.  
- **Categorical Analysis:** Higher churn rates were observed among customers who primarily use **mobile banking apps**.  

## Data Preprocessing  

To ensure high model performance, the following preprocessing steps were applied:  

- **Handling Duplicates:** 284 duplicate records were removed.  
- **Missing Values Treatment:** Categorical missing values were assigned "None"; unresolved cases were marked "No Issue."  
- **Feature Encoding:**  
  - One-hot encoding for nominal categorical variables.  
  - Ordinal encoding for income level.  
- **Scaling:** Standardized numerical features.  
- **Class Imbalance Handling:** **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the dataset.  
- **Feature Selection:** **Recursive Feature Elimination (RFECV)** identified **19 optimal features** for modeling.  

## Model Development  

### **1. K-Nearest Neighbors (KNN) - Python**  

- **Goal:** Classify customers as churned or retained based on similarity to others.  
- **Hyperparameter Tuning:** determine the optimal `k` value.  
- **Performance:** Achieved an **F1-score of 75%**.  

### **2. Random Forest - Python**  

- **Goal:** Improve predictive performance and interpretability.  
- **Hyperparameters Tuned:**  
  - `n_estimators`: 250–260 trees  
  - `max_depth`: 10–20  
- **Performance:** Best-performing model with an **F1-score of 84%**.  
- **Feature Importance:** Identified **login frequency, age, and income level** as key churn predictors.  


## Model Implementation in **Azure ML**  

To compare performance, **Random Forest and ANN models were implemented in Azure ML Studio**.  

### **Azure ML Pipeline Steps**  

1. **Data Preprocessing**  
   - Standardized numerical features using **Z-score normalization**.  
   - Categorical features were **converted to indicators (one-hot encoding)**.  
   - **SMOTE oversampling** was applied to address class imbalance.  

2. **Training Models in Azure ML**  
   - **Two-Class Decision Forest (Random Forest Equivalent)**
     - **Parameters:**  
       - Number of trees: **8**  
       - Maximum depth: **32**  
       - Random state: **123**  
     - **Performance:** F1-score of **83%**  
   - **Two-Class Neural Network (ANN)**
     - **Parameters:**  
       - Hidden layers: **100**  
       - Learning rate: **0.1**  
       - Iterations: **100**  
       - Random state: **123**  
     - **Performance:** F1-score of **48%**  

### **Python vs. Azure ML Performance Comparison**  

| Model | Platform | Accuracy | Precision | Recall | F1-Score |
|--------|------------|----------|-----------|--------|----------|
| KNN | Python | 90% | 75% | 74% | 75% |
| Random Forest | Python | **95%** | **96%** | 76% | **84%** |
| Two-Class Decision Forest | Azure ML | 94% | 88% | **78%** | 83% |
| Two-Class Neural Network | Azure ML | 82% | 54% | 43% | 48% |

- **Random Forest (Python) outperformed all models**.  
- **Azure ML’s Two-Class Decision Forest performed comparably** to Python’s Random Forest.  
- **ANN underperformed in both environments**, suggesting deep learning may not be ideal for this dataset.  

## Business Benefits & Impact  

By implementing a **customer churn prediction system**, banks can:  

- **Retain customers** by identifying at-risk individuals early.  
- **Allocate resources efficiently** by focusing on major churn drivers.  
- **Enhance customer engagement** through personalized retention strategies.  
- **Optimize operations** by addressing churn-prone customer segments.  

## Actionable Recommendations  

- **Increase Customer Engagement**:  
  - Implement **personalized emails & push notifications** for low-login customers.  
- **Targeted Retention Strategies**:  
  - Offer **age-based engagement initiatives** (e.g., social media for younger users).  
- **Deploy Real-Time Churn Monitoring**:  
  - Use the trained **Random Forest model** in production to flag at-risk customers.  
- **Improve Digital Banking Services**:  
  - Enhance **mobile banking experience** to reduce app-related churn.  

## How to Run  

### Clone the Repository  

```sh
git clone https://github.com/kenny-balogun/Bank_customer_churn.git
```
### Install Dependencies

```sh
pip install -r requirements.txt
```
### Run the Model

```sh
jupyter notebook customer_churn.ipynb
```
