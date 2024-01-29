# LOAN APPROVAL FORECASTING

<img height=350 alt="Repo Banner - Awesome Repo Template" src="https://capsule-render.vercel.app/api?type=waving&height=300&color=6FC7E1&text=Loan%20Approval%20Forecasting&fontSize=60&textBg=false&section=header&animation=twinkling&fontColor=FFFFFF&reversal=false"></img>

<p align='center'>
  <img src="https://img.shields.io/badge/3.11.0-%2329A4D1?style=flat-square&label=python">
  <img src="https://img.shields.io/badge/Binary%20classification-%23E2A24F?style=flat-square">
  <a href="https://loanapprovalforecasting-by-dysff-ei9kbuafdtgxyhadfdwuum.streamlit.app">
  <img src=https://img.shields.io/badge/OPEN%20ON%20STREAMLIT-black?style=flat-square>
  </a>
  <a href="https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data">
  <img src=https://img.shields.io/badge/dataset-blue?style=flat-square>
  </a>
</p>

## Problem this project solves

The application uses a trained machine learning model to predict the issuance of a loan to a person. The model is deployed using Streamlit. The app should help people determine whether they will be eligible for a loan or not.

## Technologies used

<p align='center'>
  <img src="https://img.shields.io/badge/numpy-black?style=flat-square&logo=numpy">
  <img src="https://img.shields.io/badge/pandas-black?style=flat-square&logo=pandas">
  <img src="https://img.shields.io/badge/streamlit-black?style=flat-square&logo=streamlit">
  <img src="https://img.shields.io/badge/scikit%20learn-black?style=flat-square&logo=scikit-learn">
  <img src="https://img.shields.io/badge/joblib-black?style=flat-square&logo=joblib">
</p>

## Features
- Project info:
  <details><summary>Metrics</summary>
  <b>METRIC: RECALL</b> 
  It is required to identify all unreliable clients. If the prediction is not clear, then it is worth rejecting this client in order to avoid subsequent problems with him. It is for this reason that the best metric for this would be recall.
  </details>
  <details><summary>Methods</summary>
    <ul>
      <li>Object-Oriented Programming (OOP)</li>
      <li>Weight of Evidence(WOE), Information Value(IV)</li>
      <li>IQR</li>
      <li>Data Analysis</li>
      <li>Data Visualization</li>
      <li>Data Cleaning</li>
      <li>Feature Engineering</li>
      <li>User Interface Design</li>
      <li>Model Deployment using Streamlit</li>
      <li>Integration of User Inputs</li>
      <li>Data Upload Mechanism</li>
      <li>New Data Handling in the Web App</li>
    </ul>
  </details>
  <details><summary>Problems encountered in the process</summary>
  <b>SMOTE</b> 
  <p>
  As you know, SMOTE is not recommended to be applied to categorical data. It uses KNN to create new samples to solve the problem of data imbalance. Since 90% of dataset is categorical data, using this method will lead to anomalies in the new samples created by SMOTE. Basically, getting numbers other than 1 or 0 in binary columns, getting numeric values in several columns of one hot encoded column at once. All this leads to the fact that a mixture of abnormal and normal data gets into the training data, on which the model is then trained. The model may have good performance when evaluated on training data, but in fact, when predicting on new data(data entered by the user), the model's performance will be many times worse.
  </p>
  <p>
  To deal with this problem, I used the compute_sample_weight method from the scikit-learn library. With SMOTE method, recall was 0.91, and after weights sampling, model's recall is 0.80.
  </p>

## Example and Usage
You can check how this app works by yourself, just click on **OPEN ON STREAMLIT** badge.

![Streamlitgif](assets/screen-capture.gif)

## License <a href="LICENSE"> ![GitHub](https://img.shields.io/github/license/MarketingPipeline/Awesome-Repo-Template) </a>

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE) file for
details.