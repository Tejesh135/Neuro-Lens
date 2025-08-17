import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import numpy as np

# Example: load your training data (replace with your actual path or data loading)
data = pd.read_csv(r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv")


# Separate target variable from features
target = "Depression"  # change as per your actual target column name
y = data[target]
X = data.drop(columns=[target])

# Define numeric and categorical columns as per your features
numeric_features = ['Age', 'CGPA', 'Work/Study Hours']
categorical_features = [
    "Gender", "City", "Profession", "Degree",
    "Dietary Habits", "Sleep Duration",
    "Have you ever had suicidal thoughts ?", "Family History of Mental Illness",
    "Academic Pressure", "Work Pressure",
    "Study Satisfaction", "Job Satisfaction", "Financial Stress"
]

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess features
X_processed = preprocessor.fit_transform(X)

# Save the preprocessor for later use in deployment
joblib.dump(preprocessor, 'preprocessor.save')

# Build a simple MLP model for example
input_dim = X_processed.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop])

# Save the trained model
model.save('mlp_model.h5')

print("Preprocessor and model retrained and saved successfully.")