import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path, na_values=['?'])

sns.set(style="whitegrid")

# 1. Most common cities
plt.figure(figsize=(8,5))
city_counts = df['City'].value_counts().head(10)
sns.barplot(x=city_counts.values, y=city_counts.index, palette='deep')
plt.title('Top Cities in Dataset')
plt.xlabel('Count')
plt.ylabel('City')
plt.tight_layout()
plt.show()

# 2. Most common degrees
plt.figure(figsize=(8,5))
degree_counts = df['Degree'].value_counts().head(10)
sns.barplot(x=degree_counts.values, y=degree_counts.index, palette='tab20')
plt.title('Top Degrees in Dataset')
plt.xlabel('Count')
plt.ylabel('Degree')
plt.tight_layout()
plt.show()

# 3. Dietary habits distribution
plt.figure(figsize=(8,5))
diet_counts = df['Dietary Habits'].value_counts()
sns.barplot(x=diet_counts.values, y=diet_counts.index, palette='muted')
plt.title('Dietary Habits Distribution')
plt.xlabel('Count')
plt.ylabel('Dietary Habits')
plt.tight_layout()
plt.show()

# 4. Study Satisfaction Distribution (sentiment proxy)
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Study Satisfaction', hue='Depression', bins=10, kde=True, palette='Set1')
plt.title('Study Satisfaction by Depression')
plt.xlabel('Study Satisfaction')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 5. Suicidal Thoughts vs Depression
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Have you ever had suicidal thoughts ?', hue='Depression', palette='Set2')
plt.title('Suicidal Thoughts vs Depression')
plt.xlabel('Suicidal Thoughts (Yes/No)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 6. Sleep Duration vs Depression
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Sleep Duration', hue='Depression', palette='Spectral')
plt.title('Sleep Duration vs Depression')
plt.xlabel('Sleep Duration')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 7. Work/Study Hours vs Depression
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Work/Study Hours', hue='Depression', bins=10, palette='cool')
plt.title('Work/Study Hours vs Depression')
plt.xlabel('Work/Study Hours')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 8. Age group analysis
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 22, 30, 40, 100], labels=['Teen', 'Young Adult', 'Adult', 'Senior'])
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='AgeGroup', hue='Depression', palette='RdBu')
plt.title('Depression by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.tight_layout()
plt.show()