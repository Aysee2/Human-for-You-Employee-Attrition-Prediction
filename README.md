# Human for You – Employee Attrition Prediction

## Overview

Human for You is a Machine Learning project focused on analyzing and predicting **employee attrition**.

The objective is to identify employees who may be at risk of leaving the company and to understand the factors that have the greatest influence on employee turnover.

The project compares two supervised classification models:

* Logistic Regression
* Decision Tree

In addition to predictive performance, the project also considers the ethical implications of using Machine Learning for human resources decision-making.

## Dataset

The analysis combines several sources of employee data:

* `general_data.csv` – general employee information
* `employee_survey_data.csv` – employee survey responses
* `manager_survey_data.csv` – manager evaluations
* `in_out_time.zip` – employee arrival and departure time data (`in_time.csv` and `out_time.csv`)

These datasets are combined and processed to build a consolidated dataset for the Machine Learning models.

The target variable is **Attrition**, indicating whether an employee has left the company (`Yes` / `No`).

## Data Preprocessing

Before training the models, several preprocessing steps are performed:

* Handling missing values
* Separation of numerical and categorical variables
* Median imputation for numerical features
* Most-frequent-value imputation for categorical features
* Robust scaling of numerical features
* One-hot encoding of categorical features
* Train, validation and test dataset splitting

The preprocessing workflow is implemented using Scikit-learn pipelines and `ColumnTransformer`.

## Machine Learning Models

### Logistic Regression

The first model uses a Logistic Regression classifier with balanced class weights.

The model is evaluated using several classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

A custom classification threshold is also tested to improve the detection of employees at risk of leaving.

The analysis additionally examines the coefficients of the Logistic Regression model to identify factors associated with a higher or lower probability of attrition.

### Decision Tree

The second approach uses a Decision Tree classifier.

Parameters such as tree depth, minimum samples per split and minimum samples per leaf are controlled to limit overfitting.

The model is evaluated using the same metrics as the Logistic Regression model.

The trained decision tree is also visualized to better understand the decisions made by the model and the variables influencing employee attrition.

## Model Comparison

The `ComparaisonModele.ipynb` notebook compares the two trained models using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* ROC curves

Overall, both models achieve relatively similar performance.

The **Decision Tree** provides slightly better overall accuracy and F1-score, while the **Logistic Regression** achieves a higher recall for employees belonging to the attrition class, making it particularly useful when the priority is identifying as many employees at risk of leaving as possible.

## Employee Attrition Analysis

Beyond prediction, the project investigates the variables that have the greatest influence on employee turnover.

The analysis highlights several factors associated with attrition, including employee characteristics, business travel, job roles, career progression and previous professional experience.

These results can help identify potential areas where employee retention strategies could be considered.

## Ethical Considerations

Using Machine Learning in Human Resources raises important ethical questions.

Predictions about employee behavior should not be used as automatic decision-making tools. Model predictions may reflect biases present in historical or organizational data and should therefore be interpreted carefully.

The project includes a dedicated ethical analysis discussing the limitations, risks and responsible use of predictive models in an HR context.

Additional documentation is available in the `docs/` directory.

## Technologies

The project was developed in Python using:

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Missingno
* Joblib

## Project Structure

```text
Human-for-You/
│
├── README.md
│
├── ModeleRegression.ipynb
├── ModeleDecisionTree.ipynb
├── ComparaisonModele.ipynb
│
├── general_data.csv
├── employee_survey_data.csv
├── manager_survey_data.csv
├── in_out_time.zip
│
└── docs/
    ├── Ethical_Analysis.pdf
    └── Bibliography.pdf
```

## Running the Project

The notebooks should be executed in the following order:

1. `ModeleRegression.ipynb`
2. `ModeleDecisionTree.ipynb`
3. `ComparaisonModele.ipynb`

The first two notebooks train and save the Machine Learning models. They generate the files required by the comparison notebook:

```text
modele_logistique.pkl
modele_arbre.pkl
X_test.pkl
y_test.pkl
```

The comparison notebook then loads these files to evaluate and compare both approaches.
