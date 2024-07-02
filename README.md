# Diabetes Detection ML App

## Overview

Diabetes is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. Symptoms of high blood sugar include frequent urination, increased thirst, and increased hunger. If untreated, diabetes can lead to several complications, including diabetic ketoacidosis, hyperosmolar hyperglycemic state, cardiovascular disease, stroke, chronic kidney disease, foot ulcers, eye damage, or even death.

This dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, aims to diagnostically predict whether a patient has diabetes based on certain diagnostic measurements. All patients in this dataset are females of at least 21 years old from Pima Indian heritage.

## Objective

Develop a machine learning model to accurately predict whether or not the patients in the dataset have diabetes.

## Dataset Details

The dataset includes several medical predictor variables and one target variable, Outcome. The predictor variables are:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)

The target variable is:
- **Outcome**: Class variable (0 or 1)

Number of observations: 768  
Number of variables: 9  

## Result

The model created as a result of XGBoost hyperparameter optimization achieved the lowest Cross Validation Score value of 0.90.

## Demo

Check out the demo of the application [here](google.com).

## Skills Demonstrated

1. Exploratory Data Analysis
2. Data Preprocessing
3. Feature Engineering
4. One Hot Encoding
5. Base Models
6. Model Tuning
7. Comparison of Final Models
8. Reporting