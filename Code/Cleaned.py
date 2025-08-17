import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path)

# View first rows
print(df.head())

# Encode categorical columns
categorical_cols = ['Gender', 'City', 'Profession', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Select features and target
X = df.drop(columns=['id', 'Depression'])
y = df['Depression']

# Scaling numeric features (optional but recommended)
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features shape:", X_train.shape)
print("Target distribution:\n", y_train.value_counts())
print("Data preprocessing completed successfully.")
# Save the cleaned data to a new CSV file
cleaned_file_path = r"C:\Users\poola\Downloads\Neuro lens\cleaned_student_depression_dataset.csv"