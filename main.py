import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 1. DATA LOADING

cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Loading datasets using your exact local directory block [cite: 1, 2, 3, 4]
cleveland = pd.read_csv("E:/PROJECT/Python/PYTHONHEART/processed.cleveland.data", names=cols)
hungarian = pd.read_csv("E:/PROJECT/Python/PYTHONHEART/processed.hungarian.data", names=cols)
switzerland = pd.read_csv("E:/PROJECT/Python/PYTHONHEART/processed.switzerland.data", names=cols)
va = pd.read_csv("E:/PROJECT/Python/PYTHONHEART/processed.va.data", names=cols)

# Concatenating the datasets 
df = pd.concat([cleveland, hungarian, switzerland, va], ignore_index=True)

# 2. DATA PREPROCESSING & CLEANING

# Replace '?' with NaN 
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)

# Handling Missing Values (NaNs)
for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    df[col] = df[col].fillna(df[col].median())

for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fix Clinical Anomalies
df['trestbps'] = df['trestbps'].replace(0, df['trestbps'].median())
df['chol'] = df['chol'].replace(0, df['chol'].median())

# Deduplication
df = df.drop_duplicates()

# Renaming Columns 
column_map = {
    'age': 'Age', 'sex': 'Gender', 'cp': 'Chest_Pain_Type',
    'trestbps': 'Resting_BP', 'chol': 'Cholesterol', 'fbs': 'Fast_Sugar_High',
    'restecg': 'ECG_Results', 'thalach': 'Max_Heart_Rate', 'exang': 'Exercise_Angina',
    'oldpeak': 'ST_Depression', 'slope': 'ST_Slope', 'ca': 'Major_Vessels',
    'thal': 'Thalassemia_Type', 'target': 'Heart_Severity'
}
df = df.rename(columns=column_map)

# Row value mapping for visualization
df_viz = df.copy()
df_viz['Gender'] = df_viz['Gender'].map({0: 'Female', 1: 'Male'})
df_viz['Chest_Pain_Type'] = df_viz['Chest_Pain_Type'].map({
    1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic'
})
df_viz['Exercise_Angina'] = df_viz['Exercise_Angina'].map({0: 'No', 1: 'Yes'})
df_viz['Heart_Condition'] = df_viz['Heart_Severity'].apply(lambda x: 'Disease' if x > 0 else 'Healthy')

# Age grouping
df_viz['Age_Group'] = pd.cut(df_viz['Age'], bins=[0, 45, 60, 100], labels=['Young', 'Middle-Aged', 'Senior'])

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 3. 5 OBJECTIVE-BASED ANALYSES


# Objective 1: Clinical Symptom Correlation
# Analysis: How do symptoms like Chest Pain and Exercise Angina interact to predict disease?
plt.figure()
sns.heatmap(pd.crosstab(df_viz['Chest_Pain_Type'], df_viz['Exercise_Angina'], values=df['Heart_Severity'].apply(lambda x: 1 if x > 0 else 0), aggfunc='mean'), annot=True, cmap='YlOrRd')
plt.title("Objective 1: Probability of Disease by Chest Pain & Exercise Angina", fontsize=14)
plt.xlabel("Exercise Induced Angina"); plt.ylabel("Chest Pain Type")
plt.show()

# Objective 2: Cardiovascular Capacity vs Age
# Analysis: Does Heart Rate capacity decline faster in Heart Disease patients as they age?
plt.figure()
sns.lmplot(data=df_viz, x='Age', y='Max_Heart_Rate', hue='Heart_Condition', aspect=1.5, scatter_kws={'alpha':0.4})
plt.title("Objective 2: Age vs Max Heart Rate Decline Trends", fontsize=14)
plt.xlabel("Age (Years)"); plt.ylabel("Max Heart Rate (BPM)")
plt.show()

# Objective 3: The Impact of Physiological Stress
# Analysis: Comparing the combined impact of ST Depression and Cholesterol.
plt.figure()
sns.scatterplot(data=df_viz, x='ST_Depression', y='Cholesterol', hue='Heart_Condition', palette='coolwarm', alpha=0.6)
plt.title("Objective 3: Interaction of Ischemia (ST Depression) and Cholesterol", fontsize=14)
plt.xlabel("ST Depression"); plt.ylabel("Cholesterol (mg/dl)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Objective 4: Diagnostic Value of Major Vessels
# Analysis: Does the number of major vessels blocked (CA) correlate with disease severity?
plt.figure()
sns.boxenplot(data=df_viz, x='Major_Vessels', y='Age', hue='Heart_Condition')
plt.title("Objective 4: Blocked Major Vessels vs Age & Disease Status", fontsize=14)
plt.xlabel("Number of Major Vessels"); plt.ylabel("Age")
plt.show()

# Objective 5: Fasting Blood Sugar and Gender Risk
# Analysis: Does high Fasting Blood Sugar carry a higher risk for one gender over the other?
plt.figure()
sns.barplot(data=df_viz, x='Gender', y=df['Heart_Severity'].apply(lambda x: 1 if x > 0 else 0), hue='Fast_Sugar_High', palette='viridis')
plt.title("Objective 5: Gender-wise Disease Risk based on Fasting Blood Sugar", fontsize=14)
plt.xlabel("Gender"); plt.ylabel("Probability of Heart Disease")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# 4. MACHINE LEARNING MODEL IMPLEMENTATION 
print("\n--- Starting Machine Learning Implementation ---")

# Prepare Data
X = df.drop(['Heart_Severity'], axis=1)
y = df['Heart_Severity'].apply(lambda x: 1 if x > 0 else 0) # Binary Classification

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Machine Learning: Confusion Matrix (Heart Disease Prediction)", fontsize=14)
plt.xlabel("Predicted Status (0: Healthy, 1: Disease)"); plt.ylabel("Actual Status")
plt.show()

# Feature Importance
plt.figure()
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title("Machine Learning: Feature Importance for Prediction", fontsize=14)
plt.xlabel("Importance Score"); plt.ylabel("Clinical Attribute")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# 5. VISUALIZATIONS 

# 1. Age Distribution
plt.figure()
plt.hist(df['Age'], bins=20, color='purple', edgecolor='black', alpha=0.8)
plt.title("Distribution of Patient Age", fontsize=14)
plt.xlabel("Age (Years)"); plt.ylabel("Number of Patients")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Gender vs Heart Health
plt.figure()
sns.countplot(data=df_viz, x='Gender', hue='Heart_Condition', palette='coolwarm')
plt.title("Heart Health Comparison by Gender", fontsize=14)
plt.xlabel("Gender"); plt.ylabel("Total Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 3. Chest Pain Symptoms vs Health
plt.figure()
sns.countplot(data=df_viz, x='Chest_Pain_Type', hue='Heart_Condition', palette='magma')
plt.title("Chest Pain Category vs Heart Condition", fontsize=14)
plt.xlabel("Symptom Type"); plt.ylabel("Patient Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 4. Cholesterol Levels Boxplot
plt.figure()
sns.boxplot(data=df_viz, x='Heart_Condition', y='Cholesterol', palette='Set2')
plt.title("Cholesterol Levels in Healthy vs. Diseased Patients", fontsize=14)
plt.xlabel("Heart Status"); plt.ylabel("Cholesterol (mg/dl)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 5. Max Heart Rate Achievement Density
plt.figure()
sns.kdeplot(data=df_viz, x='Max_Heart_Rate', hue='Heart_Condition', fill=True, alpha=0.5)
plt.title("Cardiovascular Capacity (Max Heart Rate)", fontsize=14)
plt.xlabel("Beats Per Minute (BPM)"); plt.ylabel("Density")
plt.grid(linestyle=':', alpha=0.5)
plt.show()

# 6. ST Depression Variations
plt.figure()
sns.violinplot(data=df_viz, x='Heart_Condition', y='ST_Depression', palette='muted', split=True)
plt.title("Variations in ST Depression (Ischemia Marker)", fontsize=14)
plt.xlabel("Cardiac Health Status"); plt.ylabel("ST Depression Value")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='RdYlGn', cbar=True)
plt.title("Clinical Feature Correlation Matrix", fontsize=14)
plt.xlabel("Medical Attributes"); plt.ylabel("Medical Attributes")
plt.show()

# 8. Age vs. Peak Heart Rate Scatterplot
plt.figure()
sns.scatterplot(data=df_viz, x='Age', y='Max_Heart_Rate', hue='Heart_Condition', alpha=0.7, palette='husl')
plt.title("Relationship Between Age and Peak Heart Rate", fontsize=14)
plt.xlabel("Patient Age (Years)"); plt.ylabel("Max Heart Rate (BPM)")
plt.grid(linestyle='--', alpha=0.5)
plt.show()

# 9. Exercise Induced Angina Probability Map
plt.figure()
risk_map = pd.crosstab(df_viz['Exercise_Angina'], df_viz['Heart_Condition'], normalize='index')
sns.heatmap(risk_map, annot=True, cmap='Reds', cbar=False)
plt.title("Risk Heatmap: Exercise Induced Angina Influence", fontsize=14)
plt.xlabel("Heart Health Status"); plt.ylabel("Angina During Exercise (Yes/No)")
plt.show()

# 10. Major Vessel Blockage count
plt.figure()
sns.countplot(data=df_viz, x='Major_Vessels', hue='Heart_Condition', palette='viridis')
plt.title("Blocked Vessels Impact on Disease Status", fontsize=14)
plt.xlabel("Number of Major Vessels (0-3)"); plt.ylabel("Total Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 11. Resting BP Progression with Age
plt.figure()
sns.regplot(data=df, x='Age', y='Resting_BP', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Trend: Resting Blood Pressure vs. Patient Age", fontsize=14)
plt.xlabel("Age (Years)"); plt.ylabel("Resting BP (mm Hg)")
plt.grid(linestyle='--', alpha=0.5)
plt.show()


# 12. Life Stage Group Prevalence
plt.figure()
sns.countplot(data=df_viz, x='Age_Group', hue='Heart_Condition', palette='spring')
plt.title("Heart Disease Occurrence Across Age Groups", fontsize=14)
plt.xlabel("Age Category"); plt.ylabel("Total Patients")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 13. Impact of Exercise ST Slope
plt.figure()
sns.countplot(data=df_viz, x='ST_Slope', hue='Heart_Condition', palette='Set1')
plt.title("Peak Exercise ST Slope vs Cardiac Health", fontsize=14)
plt.xlabel("ST Slope Type"); plt.ylabel("Patient Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 14. Risk Indicators Interaction (Pairplot)
g = sns.pairplot(df_viz[['Age', 'Max_Heart_Rate', 'ST_Depression', 'Heart_Condition']], hue='Heart_Condition', palette='husl')
g.fig.suptitle("Inter-correlation of Core Risk Indicators", y=1.02, fontsize=14)
for ax in g.axes.flatten():
    ax.grid(linestyle=':', alpha=0.5)
plt.show()
