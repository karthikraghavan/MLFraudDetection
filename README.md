# **Bank Account Fraud Detection**

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

1. Raw Data accuracy outperforms Transformed data
2. Weak precision and recall for fraudulent accounts
3. Transformed data shows better precision and recall

Confusion matrix for Raw dataset

<img width="175" alt="image" src="https://github.com/user-attachments/assets/d43a6c7d-08e1-446a-af16-f14afba77b6e">

Confusion matrix for Transformed dataset

<img width="186" alt="image" src="https://github.com/user-attachments/assets/12d46cf1-bd18-40d8-b1f7-fb9076697727">


#### Anamoly Detection

1. Raw Data accuracy outperforms Transformed data
2. Weak precision and recall for fraudulent accounts
3. Transformed data shows better precision and recall
  
Precision
   
<img width="176" alt="image" src="https://github.com/user-attachments/assets/fa1ffac1-27ab-4f3f-a7c7-49ac845e8988">

Recall

<img width="177" alt="image" src="https://github.com/user-attachments/assets/263dad5d-55c8-4921-a11d-9d224ceaa9d0">

Ensemble Learning Analysis

1. The model has high accuracy on both training and test sets
2. Zero precision and recall for fraudulent accounts
3. Transformed data shows better precision and recall

KFold Validation Learning Curve for raw

<img width="176" alt="image" src="https://github.com/user-attachments/assets/804f5f7c-c6f1-489a-805d-c0e5129b3d00">

KFold validation for Transformed

<img width="168" alt="image" src="https://github.com/user-attachments/assets/32cd83bf-b07d-4c50-9798-6abb5134a965">


## Conclusion

Based on all algorithms' accuracy scores and classification reports, the dataset's quality plays a significant role in developing a robust prediction model.
1.	The dataset should have a balanced distribution of classes or outcomes. The BAF dataset is highly imbalanced, which lead to biased models favoring legitimate accounts and performing poorly on fraudulent accounts.
2.	The data should be accurate, consistent, and free from errors or outliers. Many features of the BAF dataset had a skewed distribution due to outliers that negatively impacted the model's performance and reliability.
3.	The dataset should capture the variability and diversity present in real-world scenarios. Many features have imbalances that prevented the model from learning robust patterns and generalizing well to fraudulent accounts.
4.	The presence of highly correlated features in the BAF dataset leads to issues like redundancy or multi-collinearity. Removing them improved the model's performance.
5.	K-Means, Ensemble learning and One class SVM are outperformers with decent accuracy and better ability to classify both fraudulent and legitimate accounts.


### Recommendations

1.	Leveraging the complete variant of the dataset with 1M records can have drastic improvement in performance of algorithms like Anamoly detection.
2.	Explore deep learning models and neural network architectures. Time series would be most relevant for real time fraud detection
3.	Consider simpler, more interpretable models or use tools like SHAP or LIME to understand and improve complex models.
4.	Use parallel computation or GPUs to speed up model training and hyperparameter tuning.











   












