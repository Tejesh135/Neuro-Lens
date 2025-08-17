import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import optuna

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Load and preprocess your data ---
file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path, na_values=["?"])

categorical_cols = [
    'Gender', 'City', 'Profession', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
    'Sleep Duration', 'Work/Study Hours'
]
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['id', 'Depression', 'Comment']]

df_cat = pd.get_dummies(df[categorical_cols].astype(str))
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

X_tabular = pd.concat([df_num, df_cat], axis=1)
X_tabular = X_tabular.astype(np.float32).values
y = df['Depression'].astype(np.float32).values

# -------- 1. Optuna for Random Forest --------

def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        random_state=42
    )
    score = cross_val_score(rf, X_tabular, y, cv=5, scoring='accuracy').mean()
    return score

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=25, timeout=600)
print("Best RF params:", rf_study.best_params)
print("Best RF accuracy:", rf_study.best_value)

# -------- 2. Optuna for Keras MLP --------

def mlp_objective(trial):
    units1 = trial.suggest_int('units1', 32, 256)
    units2 = trial.suggest_int('units2', 16, 128)
    dropout1 = trial.suggest_float('dropout1', 0.1, 0.4)
    dropout2 = trial.suggest_float('dropout2', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = 15

    X_train, X_valid, y_train, y_valid = train_test_split(X_tabular, y, stratify=y, test_size=0.2, random_state=trial.number)

    model = tf.keras.Sequential([
        layers.Input(shape=(X_tabular.shape[1],)),
        layers.Dense(units1, activation='relu'),
        layers.Dropout(dropout1),
        layers.Dense(units2, activation='relu'),
        layers.Dropout(dropout2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_valid, y_valid), verbose=0)
    val_acc = history.history['val_accuracy'][-1]
    return val_acc

mlp_study = optuna.create_study(direction='maximize')
mlp_study.optimize(mlp_objective, n_trials=25, timeout=1200)
print("Best MLP params:", mlp_study.best_params)
print("Best MLP val accuracy:", mlp_study.best_value)