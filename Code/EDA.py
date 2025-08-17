import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path, na_values=['?'])

# Basic overview
print("Dataframe shape:", df.shape)
print("\nColumn names:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe(include='all'))

# Distribution of target variable: Depression (0=No, 1=Yes)
print("\nDepression value counts:")
print(df['Depression'].value_counts())

sns.countplot(data=df, x='Depression')
plt.title('Depression Case Distribution')
plt.show()

# Categorical variable analysis: Suicidal thoughts vs Depression
sns.countplot(data=df, x='Have you ever had suicidal thoughts ?', hue='Depression')
plt.title('Suicidal Thoughts vs Depression')
plt.show()

# Gender vs Depression
sns.countplot(data=df, x='Gender', hue='Depression')
plt.title('Gender vs Depression')
plt.show()

# Numeric variable analysis: Age distribution vs Depression
sns.histplot(data=df, x='Age', hue='Depression', kde=True, bins=20)
plt.title('Age Distribution by Depression Status')
plt.show()

# CGPA distribution by depression
sns.histplot(data=df, x='CGPA', hue='Depression', kde=True, bins=20)
plt.title('CGPA Distribution by Depression Status')
plt.show()

# Sleep Duration vs Depression
sns.countplot(data=df, x='Sleep Duration', hue='Depression')
plt.title('Sleep Duration vs Depression')
plt.show()

# Work/Study Hours vs Depression
sns.histplot(data=df, x='Work/Study Hours', hue='Depression', bins=10, kde=False)
plt.title('Work/Study Hours vs Depression')
plt.show()

# Correlation heatmap (convert Yes/No to binary for relevant columns)
df_corr = df.copy()
df_corr['Suicidal_Thoughts_Binary'] = df_corr['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
df_corr['Family_History_Binary'] = df_corr['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

# Select numeric cols and new binary cols only for correlation
corr_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 
             'Financial Stress', 'Total_Pressure', 'Sleep Duration', 'Work/Study Hours',
             'Suicidal_Thoughts_Binary', 'Family_History_Binary', 'Depression']

# If total pressure not present, create it
if 'Total_Pressure' not in df_corr.columns:
    df_corr['Total_Pressure'] = df_corr['Academic Pressure'] + df_corr['Work Pressure']

plt.figure(figsize=(12,10))
sns.heatmap(df_corr[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features Including Depression')
plt.show()