import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay)
import joblib
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
file_path = r"C:\Users\poola\Downloads\Neuro lens\student_depression_dataset.csv"
df = pd.read_csv(file_path, na_values=["?"])
X = df.drop(columns=['id', 'Depression'])
y = df['Depression']

categorical_cols = ['Gender', 'City', 'Profession', 'Dietary Habits', 'Degree',
                    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
                    'Sleep Duration', 'Work/Study Hours']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                   ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_cols),
                                 ('cat', categorical_transformer, categorical_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit random forest with grid search
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
rf_grid.fit(X_train, y_train)
joblib.dump(rf_grid.best_estimator_, r"C:\Users\poola\Downloads\Neuro lens\rf_model.pkl")
print("Random Forest model saved.")

y_pred_rf = rf_grid.predict(X_test)
y_proba_rf = rf_grid.predict_proba(X_test)[:, 1]
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()
RocCurveDisplay.from_predictions(y_test, y_proba_rf)
plt.title("Random Forest ROC Curve")
plt.show()

# Feature importance for RF
rf_best = rf_grid.best_estimator_.named_steps['classifier']
preprocess_fit = rf_grid.best_estimator_.named_steps['preprocessor']
cat_features = preprocess_fit.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_cols, cat_features])
importances = rf_best.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nTop 10 Random Forest feature importances:\n", feat_imp_df.head(10))

# Fit XGBoost with grid search
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])
xgb_param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [3, 6, 10],
    'classifier__learning_rate': [0.01, 0.1],
}
xgb_grid = GridSearchCV(xgb_pipe, xgb_param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)
xgb_grid.fit(X_train, y_train)
joblib.dump(xgb_grid.best_estimator_, r"C:\Users\poola\Downloads\Neuro lens\xgb_model.pkl")
print("XGBoost model saved.")

y_pred_xgb = xgb_grid.predict(X_test)
y_proba_xgb = xgb_grid.predict_proba(X_test)[:, 1]
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=cm_xgb).plot()
plt.title("XGBoost Confusion Matrix")
plt.show()
RocCurveDisplay.from_predictions(y_test, y_proba_xgb)
plt.title("XGBoost ROC Curve")
plt.show()

# SHAP explainability for XGBoost
print("\nCalculating SHAP values for XGBoost (may take a moment)...")
X_test_preprocessed = xgb_grid.best_estimator_.named_steps['preprocessor'].transform(X_test)
explainer = shap.Explainer(xgb_grid.best_estimator_.named_steps['classifier'])
shap_values = explainer(X_test_preprocessed)
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=all_features)

# LIME explainability for Random Forest
print("\nLIME explainability for Random Forest:")
X_train_preprocessed = rf_grid.best_estimator_.named_steps['preprocessor'].transform(X_train)
X_test_preprocessed = rf_grid.best_estimator_.named_steps['preprocessor'].transform(X_test)
explainer_lime = LimeTabularExplainer(X_train_preprocessed, feature_names=all_features, class_names=['No Depression', 'Depression'], discretize_continuous=True)
idx = 0  # Index of test example to explain
exp = explainer_lime.explain_instance(X_test_preprocessed[idx], rf_grid.best_estimator_.named_steps['classifier'].predict_proba, num_features=10)
exp.show_in_notebook(show_table=True)

# Use:
print("LIME explanation for test instance:")
print(exp.as_list())
