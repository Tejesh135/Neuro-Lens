import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN info logs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Concatenate and convert features to float32
X_tabular = pd.concat([df_num, df_cat], axis=1)
X_tabular = X_tabular.astype(np.float32).values

# Convert target to float32
y = df['Depression'].astype(np.float32).values

X_train_tab, X_test_tab, y_train, y_test = train_test_split(X_tabular, y, stratify=y, test_size=0.2, random_state=42)

# --- 1. MLP Model for tabular data ---
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
print("Evaluating MLP model...")
test_loss, test_acc = mlp_model.evaluate(X_test_tab, y_test)
print(f'MLP Test Accuracy: {test_acc:.4f}')

from sklearn.metrics import classification_report
y_pred = (mlp_model.predict(X_test_tab) > 0.5).astype("int32").flatten()
print("MLP Classification Report:")
print(classification_report(y_test, y_pred))


# --- 2. LSTM for text data (if 'Comment' exists) ---
if 'Comment' in df.columns:
    texts = df['Comment'].fillna("").values
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    X_text_seq = tokenizer.texts_to_sequences(texts)
    max_len = 100
    X_text_pad = pad_sequences(X_text_seq, maxlen=max_len)

    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
        X_text_pad, y, stratify=y, test_size=0.2, random_state=42)

    lstm_model = models.Sequential([
        layers.Embedding(input_dim=5000, output_dim=128, input_length=max_len),
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Training LSTM model on text data...")
    lstm_model.fit(X_train_text, y_train_text, epochs=10, validation_split=0.1, batch_size=32)
    print("Evaluating LSTM model...")
    loss, acc = lstm_model.evaluate(X_test_text, y_test_text)
    print(f'LSTM Test Accuracy: {acc:.4f}')
    y_pred_text = (lstm_model.predict(X_test_text) > 0.5).astype("int32").flatten()
    print("LSTM Classification Report:")
    print(classification_report(y_test_text, y_pred_text))
else:
    print("No 'Comment' column found; skipping LSTM text model.")

# --- 3. Transformer (BERT) for text (if 'Comment' exists) ---
try:
    from transformers import BertTokenizer, TFBertForSequenceClassification
    import tensorflow as tf

    if 'Comment' in df.columns:
        texts = df['Comment'].fillna("").astype(str).tolist()
        labels = y.astype(int)

        tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
        max_length = 128

        encodings = tokenizer_bert(texts, truncation=True, padding=True, max_length=max_length, return_tensors='tf')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        n = int(len(texts) * 0.8)
        train_ids, test_ids = input_ids[:n], input_ids[n:]
        train_mask, test_mask = attention_mask[:n], attention_mask[n:]
        train_labels, test_labels = labels[:n], labels[n:]

        bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        bert_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        print("Training BERT model on text data...")
        bert_model.fit(
            {'input_ids': train_ids, 'attention_mask': train_mask},
            train_labels, epochs=3, batch_size=16)

        print("Evaluating BERT model...")
        bert_model.evaluate(
            {'input_ids': test_ids, 'attention_mask': test_mask}, test_labels)
    else:
        print("No 'Comment' column found; skipping BERT model.")

except ImportError:
    print("transformers library not installed; skipping BERT model.")