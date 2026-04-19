# Cardiac-EDA-Pipeline
Data doesn't just tell stories; it saves lives. This project analyzes clinical heart data to uncover "silent" risk factors. Key findings include the high diagnostic value of ST-segment depression and the significant drop in cardiovascular capacity among diseased patients.
# 🫀 Decoding Cardiovascular Risk: A Cross-Institutional Data Analysis

## 🔎 Overview
Have you ever wondered if a "silent" symptom could actually be the loudest warning for heart disease? 

[cite_start]This project involves an in-depth clinical analysis of **920 patient records** [cite: 1, 2, 3, 4] to identify the strongest predictors of heart disease. By merging datasets from multiple geographical sources (Cleveland, Hungary, Switzerland, and VA), I built a robust diagnostic pipeline to uncover patterns that are often missed in traditional medical screenings.

## 🛠️ Technical Workflow & Preprocessing
To ensure clinical accuracy and data integrity, I implemented a rigorous cleaning process:
* [cite_start]**Standardization**: Replaced missing values marked as '?' with $NaN$ for uniform handling[cite: 1, 2, 3, 4].
* [cite_start]**Advanced Imputation**: Handled missing data using the **Median** for continuous metrics (Resting BP, Cholesterol) and **Mode** for categorical attributes[cite: 1, 2, 3, 4].
* [cite_start]**Clinical Correction**: Identified and corrected biological anomalies, such as unrealistic `0` values for Blood Pressure and Cholesterol, replacing them with median values to maintain dataset reliability[cite: 1, 2, 3, 4].
* [cite_start]**Deduplication**: Removed redundant records to ensure the statistical independence of each case[cite: 1, 2, 3, 4].

## 📊 Key Clinical Observations
* [cite_start]**The "Silent" Threat**: A staggering **79% of diseased patients were asymptomatic**, meaning they felt no typical chest pain[cite: 1, 2, 3, 4].
* [cite_start]**ST-Depression (Oldpeak)**: This emerged as a powerhouse diagnostic marker; diseased patients showed an average ST depression of **1.20**, nearly triple the average of **0.42** in healthy subjects[cite: 1, 2, 3, 4].
* [cite_start]**Cardiac Capacity**: Observed a clear drop in maximum heart rate achievement; diseased patients peaked at an average of **129 BPM** vs **148 BPM** for healthy individuals[cite: 1, 2, 3, 4].
* [cite_start]**Exercise Angina**: Diseased patients were roughly **5x more likely** to experience angina during physical exertion[cite: 1, 2, 3, 4].

## 🧪 Tech Stack
* **Language**: Python
* **Libraries**: Pandas, NumPy, Matplotlib, Seaborn
* [cite_start]**Data Origin**: Merged clinical datasets (Cleveland, Hungary, Switzerland, Long Beach VA)[cite: 1, 2, 3, 4].

## 📈 Visualizations Included
This repository features 15+ descriptive visualizations, including:
* **Correlation Heatmaps**: To identify inter-dependencies between clinical features.
* **Violin Plots**: Visualizing the distribution of ST Depression across health status.
* **KDE Density Plots**: Comparing Cardiovascular Capacity (Max Heart Rate) between cohorts.
* **Regression Analysis**: Tracking the progression of Blood Pressure with age.

---
**Acknowledgment**: Special thanks to **Maneet Kaur ma'am** for the guidance and support throughout this project. ❤️
