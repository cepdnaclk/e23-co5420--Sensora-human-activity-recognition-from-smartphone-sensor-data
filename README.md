# 📱 Sensora: Human Activity Recognition from Smartphone Sensor Data

> CO5420 – Artificial Neural Networks and Deep Learning
> Department of Computer Engineering, University of Peradeniya
> Group E23/005

---

## About the Project

**Sensora** is a Human Activity Recognition (HAR) project that investigates how machine learning and deep learning techniques can be used to recognize human physical activities using smartphone sensor data.

Modern smartphones continuously collect motion data through built-in sensors such as accelerometers and gyroscopes. This project explores how effectively these sensor signals can be used to identify everyday human activities such as walking, sitting, standing, and lying down.

Our primary objective is to compare the performance of traditional machine learning approaches with deep learning models and evaluate whether end-to-end learning from raw sensor data provides advantages over handcrafted feature engineering.

---

## Project Objectives

The goals of this project are to:

* Explore and analyze smartphone sensor data for human activity recognition.
* Implement and evaluate traditional machine learning algorithms.
* Design and train deep learning architectures for activity classification.
* Compare engineered features with raw sensor-based approaches.
* Analyze the contribution of different sensor modalities.
* Benchmark model performance through Kaggle evaluation.

---

## Dataset

This project uses the **Human Activity Recognition Using Smartphones Dataset** from the **UCI Machine Learning Repository**.

### Dataset Information

| Property             | Value                     |
| -------------------- | ------------------------- |
| Total Instances      | 10,299                    |
| Number of Activities | 6                         |
| Features             | 561                       |
| Sensors              | Accelerometer & Gyroscope |
| Sampling Frequency   | 50 Hz                     |

### Activity Classes

| Label | Activity           |
| ----- | ------------------ |
| 1     | Walking            |
| 2     | Walking Upstairs   |
| 3     | Walking Downstairs |
| 4     | Sitting            |
| 5     | Standing           |
| 6     | Laying             |

The dataset provides both:

* Engineered statistical and frequency-domain features.
* Raw accelerometer and gyroscope signals for sequence-based learning.

---

## Methodology

The project is organized into four main stages.

### Stage 1 — Baseline Machine Learning Models

Traditional machine learning algorithms will be used as baseline models:

* Decision Tree
* Support Vector Machine (SVM)

---

### Stage 2 — Feedforward Neural Networks

A Feedforward Neural Network (FNN) will be trained using the engineered features to investigate whether neural networks can improve classification performance.

---

### Stage 3 — End-to-End Sequence Learning

Sequence-based deep learning models will be trained directly on raw sensor signals:

* Long Short-Term Memory Networks (LSTM)
* CNN-LSTM Hybrid Models

---

### Stage 4 — Sensor Contribution Analysis

To understand the importance of each sensor, experiments will be conducted using:

* Accelerometer data only
* Gyroscope data only
* Combined sensor data

---

## Evaluation Metrics

The models will be evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Kaggle Leaderboard Performance

---

## Repository Structure

```text
e23-co5420-sensora-human-activity-recognition/

├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── DecisionTree.ipynb
│   ├── SVM.ipynb
│   ├── FNN.ipynb
│   ├── LSTM.ipynb
│   └── CNN_LSTM.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
│
├── models/
├── experiments/
├── results/
├── kaggle/
├── report/
└── presentation/
```

---

## Team Members — Group E23/005

| Registration Number | Name             |
| ------------------- | ----------------------------------- |
| E/23/352            | N.A.Sara                            |
| E/23/384            | S.Simasa                            |
| E/23/211            | A.M.G.Madubhashinie                 |
| E/23/435            | E.S.Wickramasinghe                  |
| E/23/427            | A.S.Weerasinghe                     |

---

## Tools and Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Jupyter Notebook
* Kaggle
* GitHub

---

## Project Timeline

| Phase  | Description                           |
| ------ | ------------------------------------- |
| Week 1 | Dataset exploration and preprocessing |
| Week 2 | Baseline machine learning models      |
| Week 3 | Feedforward neural networks           |
| Week 4 | Sequence learning models              |
| Week 5 | Evaluation and Kaggle submissions     |
| Week 6 | Final report and presentation         |

---

## Expected Outcomes

By the end of this project, we expect to:

* Compare traditional machine learning and deep learning approaches for HAR.
* Evaluate the effectiveness of end-to-end learning from raw sensor signals.
* Analyze the contribution of accelerometer and gyroscope data.
* Develop a robust activity recognition pipeline.
* Achieve competitive performance in the Kaggle evaluation.

---
