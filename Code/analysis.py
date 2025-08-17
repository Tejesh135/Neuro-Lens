import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Load and preprocess data ---
file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path, na_values=["?"])

categorical_cols = [
    'Gender', 'City', 'Profession', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
    'Sleep Duration', 'Work/Study Hours'
]
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['id', 'Depression', 'Comment']]

# One-hot encode categorical columns
df_cat = pd.get_dummies(df[categorical_cols].astype(str))

# Scale numeric columns
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Combine numeric and categorical features
X_tabular = pd.concat([df_num, df_cat], axis=1)
X_tabular = X_tabular.astype(np.float32).values

# Target variable
y = df['Depression'].astype(np.float32).values

# Split data into train/test sets
X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    X_tabular, y, stratify=y, test_size=0.2, random_state=42
)

# --- Train a simple MLP ---
mlp_model = models.Sequential([
    layers.Input(shape=(X_train_tab.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training MLP model on tabular data...")
mlp_model.fit(X_train_tab, y_train, epochs=30, validation_split=0.1, batch_size=32)

# --- Predict on test set ---
y_proba = mlp_model.predict(X_test_tab).flatten()
y_pred = (y_proba > 0.5).astype(int)

# --- Confusion matrix and classification report ---
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix on Test Set')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- Error Analysis ---
feature_names = list(df_num.columns) + list(df_cat.columns)
df_test = pd.DataFrame(X_test_tab, columns=feature_names)
df_test['True_Label'] = y_test
df_test['Pred_Label'] = y_pred

false_positives = df_test[(df_test['True_Label'] == 0) & (df_test['Pred_Label'] == 1)]
false_negatives = df_test[(df_test['True_Label'] == 1) & (df_test['Pred_Label'] == 0)]

print(f"\nTotal False Positives: {len(false_positives)}")
print(f"Total False Negatives: {len(false_negatives)}")

print("\nSample False Positives:")
print(false_positives.head())

print("\nSample False Negatives:")
print(false_negatives.head())

# Feature means for misclassification groups
print("\nMean feature values for False Positives:")
print(false_positives[feature_names].mean())

print("\nMean feature values for False Negatives:")
print(false_negatives[feature_names].mean())

# Visualization of feature distributions for errors (Example: CGPA)
if 'CGPA' in feature_names:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Pred_Label', y='CGPA', data=df_test)
    plt.title('CGPA Distribution by Predicted Label')
    plt.show()