import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =====================================================================
# FORCE ABSOLUTE REPRODUCIBILITY (LOCK THE SEEDS)
# =====================================================================


def set_reproducibility_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


set_reproducibility_seed(42)

# Load and clean dataset once globally
df_ml = pd.read_excel('Gait_Analysis_Master_Report.xlsx')
df_ml.columns = df_ml.columns.str.strip()
feature_cols = ['Stride Length', 'Gait Speed',
                'Cadence', 'Double Support', 'Swing Phase']
df_ml[feature_cols] = df_ml[feature_cols].fillna(df_ml[feature_cols].mean())

# Define the 6 specific binary pairs from the 2022 reference study
binary_pairs = [
    ['HT', 'LS'],
    ['HT', 'OB'],
    ['HT', 'TD'],
    ['LS', 'OB'],
    ['LS', 'TD'],
    ['OB', 'TD']
]

# Master storage for results
all_pair_results = []

print("🚀 Starting Automated Binary Pair Evaluation Loop...\n")

for pair in binary_pairs:
    print(f"🔄 Training models for pair: {pair[0]} vs {pair[1]}...")

    # Filter for the specific pair
    df_filtered = df_ml[df_ml['Task'].isin(pair)].copy()
    X = df_filtered[feature_cols]
    y = df_filtered['Task']

    # Process labels and scaling
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_nn = to_categorical(y_train, num_classes=2)
    y_test_nn = to_categorical(y_test, num_classes=2)

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # 2. SVM
    svm = SVC(kernel='rbf', C=5.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))

    # 3. MLP
    mlp = Sequential([
        Dense(16, activation='relu', input_shape=(5,)),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])
    mlp.fit(X_train_scaled, y_train_nn, epochs=100, batch_size=4, verbose=0)
    mlp_acc = accuracy_score(y_test, np.argmax(
        mlp.predict(X_test_scaled, verbose=0), axis=1))

    # 4. 1D CNN
    X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
    cnn = Sequential([
        Conv1D(8, kernel_size=2, activation='relu', input_shape=(5, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train_cnn, y_train_nn, epochs=100, batch_size=4, verbose=0)
    cnn_acc = accuracy_score(y_test, np.argmax(
        cnn.predict(X_test_cnn, verbose=0), axis=1))

    # Store results for this pair
    all_pair_results.append({
        'Comparison Pair': f"{pair[0]} vs {pair[1]}",
        'Random Forest': rf_acc,
        'SVM': svm_acc,
        'MLP (Neural Net)': mlp_acc,
        '1D CNN': cnn_acc
    })

# Convert to DataFrame and show final table
master_results_df = pd.DataFrame(all_pair_results)
print("\n========================================================")
print("🎯 MASTER BINARY RESULTS FOR YOUR SLIDES")
print("========================================================")
print(master_results_df.to_string(index=False))

# Save directly to your repository folder
master_results_df.to_csv('gait_binary_master_results.csv', index=False)
print("\n💾 Saved master spreadsheet as: 'gait_binary_master_results.csv'")
