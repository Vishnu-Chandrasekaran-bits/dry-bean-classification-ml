# Dry Bean Classification Using Machine Learning


## 1. Project Overview
This project implements and compares five machine learning classification algorithms for identifying dry bean varieties based on their morphological characteristics.

An interactive Streamlit application is also developed to allow users to upload test data, select a trained classification model and view its evaluation results.

### Project Objectives

- Analyse the Dry Bean dataset.
- Perform data preprocessing.
- Implement five machine learning classification algorithms.
- Evaluate the models using six performance metrics.
- Compare the performance of all five models.
- Identify the best-performing classification algorithm.
- Develop an interactive Streamlit application.

## 2. Problem Statement

The objective of this project is to develop a multiclass classification system that can identify different varieties of dry beans based on their morphological measurements.

The dataset contains numerical measurements describing the physical characteristics of dry beans. Using these measurements, the classification models predict the corresponding bean variety.

Five machine learning algorithms are implemented and evaluated using the same dataset. Their performance is compared using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 3. Dataset Description

**Dataset Name:** Dry Bean Dataset  
**Dataset Source:** [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

The dataset is used for a seven-class classification problem.

### Dataset Summary

| Attribute | Details |
|---|---|
| Dataset | Dry Bean |
| Input features | 16 |
| Target variable | Class |
| Number of classes | 7 |
| Classification type | Multiclass |
| Feature types | Integer and Continuous |
| Target type | Categorical |
| Missing values | No, according to the supplied variable table |
| Total instances | 13,611 |
| Training instances | 10,888 |
| Test instances evaluated | 2,723 |


### Target Classes

1. Seker
2. Barbunya
3. Bombay
4. Cali
5. Dermason
6. Horoz
7. Sira

### Dataset Variables

| Variable Name | Role | Type | Description | Units | Missing |
|---|---|---|---|---|---|
| Area | Feature | Integer | Area of a bean zone and number of pixels within its boundaries | Pixels | No |
| Perimeter | Feature | Continuous | Bean circumference, defined as the length of its border | — | No |
| MajorAxisLength | Feature | Continuous | Distance between the ends of the longest line that can be drawn from a bean | — | No |
| MinorAxisLength | Feature | Continuous | Longest line that can be drawn from the bean perpendicular to the main axis | — | No |
| AspectRatio | Feature | Continuous | Relationship between MajorAxisLength and MinorAxisLength | — | No |
| Eccentricity | Feature | Continuous | Eccentricity of the ellipse having the same moments as the region | — | No |
| ConvexArea | Feature | Integer | Number of pixels in the smallest convex polygon containing a bean seed | — | No |
| EquivDiameter | Feature | Continuous | Diameter of a circle having the same area as the bean seed | — | No |
| Extent | Feature | Continuous | Ratio of pixels in the bounding box to the bean area | — | No |
| Solidity | Feature | Continuous | Ratio of pixels in the convex shell to those found in beans | — | No |
| Roundness | Feature | Continuous | Calculated using (4πA)/(P²) | — | No |
| Compactness | Feature | Continuous | Measures the roundness of an object | Ed/L | No |
| ShapeFactor1 | Feature | Continuous | Description not provided in the supplied variable table | — | No |
| ShapeFactor2 | Feature | Continuous | Description not provided in the supplied variable table | — | No |
| ShapeFactor3 | Feature | Continuous | Description not provided in the supplied variable table | — | No |
| ShapeFactor4 | Feature | Continuous | Description not provided in the supplied variable table | — | No |
| Class | Target | Categorical | Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira | — | No |

### Class Distribution

The following table presents the distribution of the seven bean classes in the complete Dry Bean dataset. These counts represent all 13,611 instances before the train-test split.

| Class | Number of Instances | Percentage (%) |
| :--- | ---: | ---: |
| BARBUNYA | 1,322 | 9.71 |
| BOMBAY | 522 | 3.84 |
| CALI | 1,630 | 11.98 |
| DERMASON | 3,546 | 26.05 |
| HOROZ | 1,928 | 14.17 |
| SEKER | 2,027 | 14.89 |
| SIRA | 2,636 | 19.37 |
| **Total** | **13,611** | **100.01** |

**Note:** The percentages may not total exactly 100% because of rounding. DERMASON is the most represented class, while BOMBAY is the least represented class.

## 4. Data Preprocessing and Experimental Setup

The following preprocessing steps are performed before model training:

- Load the dataset from the Excel file.
- Remove rows containing missing values using `dropna()`.
- Separate the input features and target variable (`Class`).
- Convert categorical input features into numerical representations using `pd.get_dummies()`.
- Encode target classes using `LabelEncoder()`.
- Split the dataset into training and testing sets using stratified sampling.
- Apply `StandardScaler` through a Scikit-learn Pipeline.
- Train and save the five classification models using Joblib.
- Save the test dataset as `test_data.csv` for model evaluation.

### Experimental Configuration

| Parameter | Configuration |
| :--- | :--- |
| Train-test split | 80:20 (80% training, 20% testing) |
| Random state | 42 |
| Feature scaling | StandardScaler |
| Missing-value handling | Rows containing missing values are removed using `dropna()` |
| Duplicate handling | Not explicitly performed |
| Evaluation dataset | `test_data.csv` |

The same test dataset (`test_data.csv`) is used to evaluate all five classification models. This ensures that every model is evaluated on the same set of test instances, allowing a fair and consistent comparison of their performance.

## 5. Machine Learning Models

Five machine learning classification algorithms are implemented.

### 5.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an instance belonging to a particular class.

It is used to identify relationships between the morphological features and the corresponding dry bean varieties.

### 5.2 Decision Tree Classifier

Decision Tree is a tree-based classification algorithm that makes predictions through a sequence of feature-based decisions.

It divides the dataset into smaller subsets based on selected feature conditions.

### 5.3 K-Nearest Neighbour (KNN)

K-Nearest Neighbour is a distance-based classification algorithm that predicts the class of an instance based on the classes of its nearest neighbours.

### 5.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that features follow Gaussian distributions within each class and that features are conditionally independent given the class.

### 5.5 Support Vector Machine (SVM)

Support Vector Machine is a supervised classification algorithm that identifies decision boundaries separating different classes by maximising the margin between them.

## 6. Evaluation Metrics

The five classification models are evaluated using six performance metrics.

1. **Accuracy:** Measures the proportion of correctly classified instances.
2. **AUC Score:** Measures the ability of the model to distinguish between classes.
3. **Precision:** Measures the proportion of correctly predicted positive instances.
4. **Recall:** Measures the proportion of actual positive instances correctly identified.
5. **F1 Score:** Represents the harmonic mean of Precision and Recall.
6. **Matthews Correlation Coefficient (MCC):** Measures the overall quality of classification using confusion matrix outcomes.

### Multiclass Evaluation Configuration

Since the Dry Bean dataset contains seven target classes, multiclass evaluation techniques are used.

The following configurations are implemented:

| Metric | Configuration |
|---|---|
| Accuracy | Standard multiclass accuracy |
| AUC | One-vs-Rest (OvR), Weighted Average |
| Precision | Weighted Average |
| Recall | Weighted Average |
| F1 Score | Weighted Average |
| MCC | Multiclass Matthews Correlation Coefficient |

**Implementation details:**

- Weighted averaging is used for Precision, Recall and F1 Score.
- AUC is calculated using the One-vs-Rest (OvR) strategy with weighted averaging.
- MCC is calculated using Scikit-learn's `matthews_corrcoef()` function.
- All five classifiers are evaluated using the same test dataset.


## 7. Model Performance Comparison

The following results were extracted from the Classification Model Explorer application.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9041 | 0.9789 | 0.9038 | 0.9041 | 0.9038 | 0.8841 |
| KNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

*The following results were obtained by evaluating all five trained classification models on the same test dataset containing 2,723 instances.*

## 8. Model Performance Observations

### 8.1 Logistic Regression

Logistic Regression achieved an accuracy of 0.9214 and an AUC of 0.9934.

It demonstrated strong classification performance across the seven bean varieties. Its performance was very close to SVM, indicating that a linear classification approach is effective for this dataset.

### 8.2 Decision Tree

Decision Tree achieved an accuracy of 0.9041 and an AUC of 0.9789.

Although it provided reasonable classification performance, it performed below the other models in this experiment. Its lower AUC indicates comparatively weaker class discrimination.

### 8.3 K-Nearest Neighbour

KNN achieved an accuracy of 0.9137 and an AUC of 0.9837.

It provided good classification performance and performed better than Decision Tree and Naive Bayes in terms of accuracy.

### 8.4 Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.8979 and an AUC of 0.9902.

It had the lowest accuracy among the five models, although its AUC remained high. This indicates that its ability to distinguish between classes was strong despite comparatively lower classification accuracy.

### 8.5 Support Vector Machine

SVM achieved the highest accuracy of 0.9221, AUC of 0.9937, precision of 0.9224, recall of 0.9221, F1 score of 0.9222 and MCC of 0.9059.

It demonstrated the strongest overall performance among the five classifiers, although its advantage over Logistic Regression was marginal.

### 8.6 Overall Best-Performing Model

**Selected Model: Support Vector Machine (SVM)**

SVM achieved the highest values across all six evaluation metrics in the reported comparison.

Its accuracy of 92.21% and MCC of 0.9059 indicate strong classification performance. Its AUC of 0.9937 also demonstrates excellent class discrimination.

Logistic Regression was a close competitor, achieving an accuracy of 92.14%.

Based on the reported experimental results, SVM is selected as the overall best-performing classifier for this dataset.

## 9. Streamlit Application

An interactive web application titled **Classification Model Explorer** has been developed using Streamlit.

### Application Features

- CSV test dataset upload.
- Classification model selection dropdown.
- Evaluation metric display.
- Confusion matrix visualisation.
- Classification report.
- Comparison of all five classification models.

The application loads the saved trained models and evaluates the uploaded test dataset without retraining the models. The uploaded CSV file must contain the input features and the actual target column (`Class`) to calculate evaluation metrics, generate the confusion matrix and display the classification report.

### Live Streamlit Application

https://vishnu-chandrasekaran-dry-bean-classification-ml.streamlit.app/

## 10. GitHub Repository

**Repository Name:** dry-bean-classification-ml

**GitHub Repository URL:** https://github.com/Vishnu-Chandrasekaran-bits/dry-bean-classification-ml

The repository contains the source code, saved trained models, preprocessing artifacts, test dataset, dependency file and project documentation.

### Repository Structure


```text
dry-bean-classification-ml/
│
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── dataset.xlsx
    ├── decision_tree.joblib
    ├── feature_columns.joblib
    ├── knn.joblib
    ├── label_encoder.joblib
    ├── logistic_regression.joblib
    ├── metrics_table.csv
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── train_models.py
```

## 11. Installation and Execution

### Prerequisites

- Python
- pip
- Git

### Clone the Repository

```bash
git clone https://github.com/Vishnu-Chandrasekaran-bits/dry-bean-classification-ml.git
cd dry-bean-classification-ml
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 12. Conclusion

This project implements and compares five machine learning classification algorithms for identifying seven varieties of dry beans.

The models are evaluated using six classification performance metrics to understand their effectiveness on the selected dataset.

Based on the experimental results, Support Vector Machine achieved the strongest overall performance, with Logistic Regression performing very closely.

The Streamlit application provides an interactive interface for uploading test data, selecting trained classification models and analysing their evaluation results through metrics, confusion matrices and classification reports.

## 13. References

1. Source of the Dry Bean dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/
