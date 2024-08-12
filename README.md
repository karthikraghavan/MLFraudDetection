# Bank Account Fraud Detection

Every year, financial institutions lose billions of dollars to fraud. Identity theft, fake documents, and online banking fraud are some methods fraudsters use to open accounts and steal information. One practical approach to enhancing the fraud detection technique is to leverage Machine learning/Artificial Intelligence. The project aims to conduct a statistical analysis of the Bank Account Fraud (BAF) dataset ((Bank Account Fraud Dataset Suite (NeurIPS 2022), n.d.)) and use various machine learning algorithms to predict whether a bank account is fraudulent or legitimate accurately. The key steps include analyzing and manipulating the Bank Account Fraud dataset and creating a solid prediction algorithm to classify fraudulent accounts based on different features in the bank account application.

## About the Dataset

The dataset used in this analysis is the Bank Application Fraud Dataset published at Neuro IPS 2022.

|                 |  | 
| --------------- | --------------- | 
| **Sample**     | 1 M    | 
| **Population**     | Subjects who had applied for bank account    | 
| **# of Categorical Variables**     | 16    | 
| **# of Numerical variables**     | 15    | 
| **Missing values**     | Yes    | 
| **Prediction type**     | Classification    | 

## Data Cleaning and Preparation

1. Handling missing values
2. Encoding Categorical variables
3. Scaling and Normalization
4. Handling outliers


## Exploratory Data Analysis

Exploratory data analysis consists of univariate and bivariate analyses of the dataset's quantitative and categorical features. These insights lay the groundwork for more complex analyses, including bivariate and multivariate analyses, to further explore the relationships between variables and their impact on the fraud_bool  variable. 

### Continuous Variables - Univariate analysis


<img width="507" alt="image" src="https://github.com/user-attachments/assets/76d07409-6c99-4ea2-99ab-d3bddf957be4">


### Categorical Variables - Univariate Analysis

<img width="636" alt="image" src="https://github.com/user-attachments/assets/23bfebf1-7492-458a-bcfd-8d24f8eed32b">


### Correlation Matrix
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/e94f89d5-c72b-4f29-9427-3df244dde65b">

A high correlation between the following features
1. Month and Velocity_4w
2. Velocity_4w and Velocity_24h
3. Porposed_credit_limit and credit_score

<img width="245" alt="image" src="https://github.com/user-attachments/assets/189e4982-4f1e-46d9-aae3-14377c345697">

## Feature Engineering

We used SMOTE to oversample the fraudulent accounts by 80%% of the size of the legitimate accounts.

|                 | Original Dataset | After SMOTE Sampling        |
| --------------- | --- | ----------------- |
| Legitimate accounts        | 19540  | 19540 |
| Fradulent accounts      | 56  | 15632   |



## Model Selection and Training

1. K-Means Clustering
   - Unsupervised algorithm to cluster data. 
   - The objective is to cluster data into k-clusters 
   - Assign each observation to a cluster
   - \# of Cluster = 2


2. Anamoly Detection

  - Anomalies are also known as outliers or novelties 
  - Data points significantly differ from most of the dataset. 
  - We use a Local outlier Factor, One-class SVM, and isolation forest to detect anomalies

```
  # Define parameter grid for Isolation Forest
  param_grid_lof = {
    'n_neighbors': [10, 20, 30],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1,2],
    'contamination': [0.05, 0.1]
}

# Define parameter grid for OneClassSVM
param_grid_svm = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'nu': [0.01, 0.05, 0.1, 0.5]
}

# Define parameter grid for OneClassSVM
param_grid_svm = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'nu': [0.01, 0.05, 0.1, 0.5]
}
```

#### Hyperparameter Tuning
![image](https://github.com/user-attachments/assets/20dd5c24-dc63-4984-937a-18db70400184)

#### GridSearchCV

Evaluates the model’s performance for every combination by Hyperparameters

## Model Analysis

#### Classifier Accuracies

<img width="265" alt="image" src="https://github.com/user-attachments/assets/6ad107da-707d-4272-adbb-4c3e7aeb8892">

#### K-Means clustering Analysis![image](https://github.com/user-attachments/assets/b65c8fda-2f20-477d-a19d-07600b0c8aae)

Raw Data accuracy outperforms Transformed data
Weak precision and recall for fraudulent accounts
Transformed data shows better precision and recall
![image](https://github.com/user-attachments/assets/355f8e88-a042-4752-9237-b8186e1eb8a1)










