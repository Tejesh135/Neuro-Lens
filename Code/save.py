import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

# 1. Load and preprocess data
file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path, na_values=["?"])

categorical_cols = [
    'Gender', 'City', 'Profession', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
    'Sleep Duration', 'Work/Study Hours'
]
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['id', 'Depression', 'Comment']]

# One-hot encode categoricals
df_cat = pd.get_dummies(df[categorical_cols].astype(str))
# Scale numeric columns
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Combine features (ensure order is noted for deployment)
X_tabular = pd.concat([df_num, df_cat], axis=1).astype(np.float32).values
y = df['Depression'].astype(np.float32).values

# Save the column order for deployment use
feature_columns = list(df_num.columns) + list(df_cat.columns)
pd.Series(feature_columns).to_csv('feature_columns.csv', index=False)

# Save the scaler for deployment
joblib.dump(scaler, 'scaler.save')

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tabular, y, stratify=y, test_size=0.2, random_state=42
)

# 3. Define and train the MLP model
mlp_model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# 4. Automatically save the model after training
mlp_model.save('mlp_model.h5')
print("MLP model saved as mlp_model.h5")
