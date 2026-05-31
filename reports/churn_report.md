# Telco Churn Model Evaluation

## Executive Summary

This project predicts customer churn and translates the predictions into retention priorities. The final model is useful for ranking customers by churn risk before a campaign.

The most important business reading is not "the model is correct or wrong." The useful reading is: "which customers deserve attention first, and what type of intervention should they receive?"

## Model Snapshot

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.8407 |
| Precision | 0.6445 |
| Recall | 0.5187 |
| F1-score | 0.5748 |
| True positives | 194 |
| False positives | 107 |
| False negatives | 180 |
| True negatives | 926 |

## Threshold

Current decision threshold: `0.50`

This threshold is a starting point. In a real retention workflow, I would tune it based on campaign budget, expected saved revenue, customer lifetime value, and the cost of each retention offer.

## Classification Report

```text
              precision    recall  f1-score   support

           0      0.837     0.896     0.866      1033
           1      0.645     0.519     0.575       374

    accuracy                          0.796      1407
   macro avg      0.741     0.708     0.720      1407
weighted avg      0.786     0.796     0.788      1407

```

## Retention Playbook

| Segment | Churn probability | Action |
| --- | ---: | --- |
| High risk | `>= 0.70` | Immediate outreach, contract incentive, support bundle |
| Medium risk | `0.40 - 0.69` | Nurture campaign, service review, controlled offer test |
| Low risk | `< 0.40` | Monitor; avoid unnecessary campaign spend |

## Why This Matters

False negatives are expensive because they are customers the company fails to save. False positives are also costly because the business may spend money on customers who were not going to churn. The practical solution is to use probability ranking and campaign capacity, not a hard model label alone.

## Interview-Ready Takeaway

This project shows the full data science loop: framing a business problem, building a reproducible ML pipeline, evaluating trade-offs, creating customer-level scores, and packaging the result in a demo app that a non-technical stakeholder can use.
