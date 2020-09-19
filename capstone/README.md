# Capstone Project: Customer Segmentation and Prediction

## Installation:
1. Use 'pip install' for prerequisite packages pandas, plotly, numpy, matplotlib, seaborn, tables, sklearn, skopt, xgboost, lightgbm.

## Introduction
What are the demographic features of a typical mail order customer as compared to general population? This project answers this question through
1. Unsupervised learning for customer segmentation, i.e. to find the relationship between existing customer base and general population of Germany through principla component analysis, k-means clustering. However, most of the time spent in the project was on data wrangling.
2. developing Supervised Learning model for customer acquisition prediction. Using the the knowledge on data wrangling and interesting features from unsupervised learning model, we developed and XGBoost model to find out if an individual would respond to a marketing campaign.
3. participating in a Kaggle competition for comparison of prediction model and its performance

## Files
1. The data files are not in Github repo due to project terms and conditions. Two datasets a) demographic information sample general population (891211 persons) of Germany b) Similar information for customers (191652) of a mail-order company. There were other supporting files for interpretation of the data
2. There are two major notebooks 
    - 'Arvato Project Workbook.ipynb' which does the unsupervised learning for customer segmentation after initial data exploration 
    - 'Supervised Learning.ipynb' contains the supervised learning model for customer acquisition prediction and its optmization. It also generates the files for Kaggle submission.  
    - helper.py contains python function for cleaning data, preprocessing, and visualization.

## Results

The description, methodology, and results of the project is documented in a medium article.
https://medium.com/@hritam79/creating-a-prediction-model-for-customer-acquisition-3517f538ef66

## Licensing, Authors, Acknowledgements

Thanks to Arvato Financial Solutions, Udacity, Kaggle for the project data and input.
