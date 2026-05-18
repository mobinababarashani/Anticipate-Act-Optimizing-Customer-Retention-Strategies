## 📌 Project Overview

Customer churn is a critical problem in the telecom industry, as retaining existing customers is significantly more cost-effective than acquiring new ones.

This project aims to predict customer churn using machine learning techniques and provide actionable insights to help the business reduce churn and improve customer retention strategies.


## 2.Business Problem

The company is experiencing customer attrition, leading to revenue loss. However, not all customers should be targeted equally.

The key challenge is:

> **How can we identify high-risk customers who are likely to churn and prioritize them for retention campaigns in a cost-effective way?**

This project focuses on predicting churn probability and enabling data-driven retention decisions.


## 3.Machine Learning Problem

* Problem Type: Binary Classification
* Target Variable: `Churn` (Yes/No)
* Objective: Predict the probability of a customer leaving the service

The model will be used to rank customers based on churn risk.


## 4.Business Metrics

To evaluate the real-world impact of the model, we consider:

**Churn Rate Reduction**
**Customer Retention Rate**
**Customer Lifetime Value (CLV)**

From a modeling perspective:

**Recall (for churn class)** is prioritized to avoid missing high-risk customers.


## 5.Cost Consideration

Different types of prediction errors have different business impacts:

**False Positive (FP):** Targeting a customer who would not churn → Marketing cost
**False Negative (FN):** Missing a customer who churns → Revenue loss

> In this project, minimizing False Negatives is more critical than minimizing False Positives.


## 6.Initial Hypotheses

Based on domain intuition, we hypothesize:

* Customers with short tenure are more likely to churn
* Monthly contracts lead to higher churn compared to long-term contracts
* Lack of technical support increases churn risk
* Higher monthly charges may increase churn probability

These hypotheses will be validated during EDA.


## ❓ 7. Key Questions

* Which type of customers are most likely to churn?
* What factors contribute most to churn behavior?
* Which customers should be prioritized for retention campaigns?
* What actionable strategies can reduce churn?


