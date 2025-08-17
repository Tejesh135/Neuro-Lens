import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# --- 1. Load and preprocess data ---
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

# Combine features as numpy array
X_tabular = pd.concat([df_num, df_cat], axis=1).astype(np.float32).values
y = df['Depression'].astype(np.float32).values

# --- 2. Train-test split keeping indices for fairness reference ---
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, stratify=y, test_size=0.2, random_state=42)

X_train_tab = X_tabular[train_idx]
X_test_tab = X_tabular[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
df_test_orig = df.iloc[test_idx].reset_index(drop=True)  # For group analysis

# --- 3. Train sample Keras MLP model ---
mlp_model = models.Sequential([
    layers.Input(shape=(X_train_tab.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Training MLP model on tabular data ...")
mlp_model.fit(X_train_tab, y_train, epochs=20, validation_split=0.1, batch_size=32, verbose=1)

# --- 4. Make predictions on test set ---
y_proba = mlp_model.predict(X_test_tab).flatten()
y_pred = (y_proba > 0.5).astype(int)

# --- 5. Fairness analysis preparation ---
df_test_orig['Pred_Label'] = y_pred
df_test_orig['True_Label'] = y_test

# --- 6. Function for fairness assessment by group ---
def fairness_metrics(df, group_col):
    groups = df[group_col].dropna().unique()
    records = []
    for g in groups:
        group_df = df[df[group_col] == g]
        y_true = group_df['True_Label']
        y_pred = group_df['Pred_Label']
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        records.append({
            group_col: g,
            'Count': len(group_df),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'False Positives': fp,
            'False Negatives': fn
        })
    return pd.DataFrame(records)

# --- 7. Report fairness by Gender ---
gender_fairness = fairness_metrics(df_test_orig, 'Gender')
print("\nFairness metrics by Gender:")
print(gender_fairness)

# --- 8. Report fairness by top 5 Cities ---
top_cities = df_test_orig['City'].value_counts().index[:5]
city_subset = df_test_orig[df_test_orig['City'].isin(top_cities)]
city_fairness = fairness_metrics(city_subset, 'City')
print("\nFairness metrics by City (top 5):")
print(city_fairness)

# --- 9. Report fairness by Profession ---
profession_fairness = fairness_metrics(df_test_orig, 'Profession')
print("\nFairness metrics by Profession:")
print(profession_fairness)

# --- 10. Visualization ---
for group_name, df_metrics in [('Gender', gender_fairness), ('Profession', profession_fairness)]:
    plt.figure(figsize=(10, 6))
    df_metrics.set_index(group_name)[['Accuracy', 'Precision', 'Recall', 'F1-score']].plot(kind='bar')
    plt.title(f'Fairness Metrics by {group_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.show()