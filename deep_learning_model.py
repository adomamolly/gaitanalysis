import random

import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, BatchNormalization, Layer
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Reshape
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =====================================================================
# 1. SET GLOBAL REPRODUCIBILITY SEED
# =====================================================================


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


set_seeds(42)

# =====================================================================
# 2. CONFIGURATION & DATA INGESTION
# =====================================================================
DATA_DIR = "sd"
TARGET_TASKS = ['HT', 'LS', 'OB']
MAX_TIMESTEPS = 700  #

X_list = []
y_list = []

print("📂 Scanning subject folders for raw data tracks...")

# Search through all subject directories (e.g., SE1-N01, SE1-N02...)
subject_folders = sorted(glob.glob(os.path.join(DATA_DIR, "SE1-N*")))

for folder in subject_folders:
    for task in TARGET_TASKS:
        # Match files like: sd/SE1-N01/SE1-N01-HT.txt
        file_pattern = os.path.join(folder, f"*-{task}.txt")
        matched_files = glob.glob(file_pattern)

        for file_path in matched_files:
            try:
                # Load raw sensor matrix space-delimited or tab-delimited
                # Auto-detect delimiter
                raw_matrix = np.loadtxt(file_path, skiprows=1, delimiter=None)

                # Check if file has enough data points
                if raw_matrix.shape[0] < 10:
                    continue

                # Standardize length: Truncate if too long, pad with zeros if too short
                if raw_matrix.shape[0] >= MAX_TIMESTEPS:
                    processed_series = raw_matrix[:MAX_TIMESTEPS, :]
                else:
                    padding_length = MAX_TIMESTEPS - raw_matrix.shape[0]
                    padding = np.zeros((padding_length, raw_matrix.shape[1]))
                    processed_series = np.vstack((raw_matrix, padding))

                X_list.append(processed_series)
                y_list.append(task)

            except Exception as e:
                print(
                    f"⚠️ Skipping damaged file {os.path.basename(file_path)}: {e}")


def compute_angular_differences(X_raw_data):
    """
    Transforms absolute quaternions into temporal orientation differences.
    This eliminates sign flips and subject-specific sensor alignment shifts.
    """
    # X_raw_data shape: (samples, timesteps, features)
    num_samples, num_timesteps, num_features = X_raw_data.shape
    # Initialize an array to hold the localized changes
    X_diff = np.zeros((num_samples, num_timesteps - 1, num_features))

    for s in range(num_samples):
        for t in range(num_timesteps - 1):
            # Calculate the frame-to-frame delta differential
            X_diff[s, t, :] = X_raw_data[s, t+1, :] - X_raw_data[s, t, :]

    print(
        f"🔄 Transformed raw streams into smooth kinematic deltas. New shape: {X_diff.shape}")
    return X_diff


# Convert lists to NumPy arrays
X_raw = np.array(X_list)  # Target Shape: (Samples, Timesteps, Features)
y_raw = np.array(y_list)

X_transformed = compute_angular_differences(X_raw)


print(f"\n✅ Data collection complete!")
print(f"📊 Total raw trials processed: {X_raw.shape[0]}")
print(f"⏱️ Time steps per trial: {X_raw.shape[1]}")
print(f"📉 Total input sensors tracking: {X_raw.shape[2]}")

# =====================================================================
# 3. PREPROCESSING & SPLITTING
# =====================================================================
# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded, num_classes=3)

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Flatten to 2D to apply standard scaling across all time steps uniformly
num_features = X_raw.shape[2]
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Flatten inputs specifically for the standard MLP architecture
X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

# =====================================================================
# 4. DEEP LEARNING MODEL EXECUTION
# =====================================================================

# Stops training early if validation loss stops improving for 5 epochs straight
monitor_callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

# --- MODEL 1: MULTILAYER PERCEPTRON (MLP) ---
print("\n🧠 Training Multilayer Perceptron (MLP)...")
mlp = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_flat.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
mlp.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
mlp.fit(X_train_flat, y_train, epochs=150, batch_size=8,
        validation_data=(X_test_flat, y_test), callbacks=monitor_callbacks, verbose=1)
_, mlp_acc = mlp.evaluate(X_test_flat, y_test, verbose=0)

# --- MODEL 2: 2D CONVOLUTIONAL NEURAL NETWORK (2D CNN) ---
print("\n🌀 Training 2D CNN")
cnn = Sequential([
    Conv2D(32, kernel_size=(5, 3), activation='relu',
           input_shape=(MAX_TIMESTEPS, 40, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')  # 3-class output for HT, LS, OB
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(X_train_scaled, y_train, epochs=150, batch_size=16,
        validation_data=(X_test_scaled, y_test), verbose=1)
_, cnn_acc = cnn.evaluate(X_test_scaled, y_test, verbose=0)

# # --- MODEL 3: LONG SHORT-TERM MEMORY NETWORK (LSTM) ---
print("\n⏳ Training LSTM (Temporal Sequence Extractor)...")
lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=False),
                  input_shape=(MAX_TIMESTEPS, num_features)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
lstm.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])
lstm.fit(X_train_scaled, y_train, epochs=150, batch_size=8,
         validation_data=(X_test_scaled, y_test), callbacks=monitor_callbacks, verbose=1)
_, lstm_acc = lstm.evaluate(X_test_scaled, y_test, verbose=0)

# =====================================================================
# 5. FINAL RESULTS SUMMARY
# =====================================================================
print("\n========================================================")
print("🎯 RAW TIME-SERIES DEEP LEARNING BENCHMARKS (3-CLASS)")
print("========================================================")
print(f"🌀 2D CNN Architecture Accuracy   : {cnn_acc * 100:.2f}%")
print(f"🤖 Multilayer Perceptron Accuracy : {mlp_acc * 100:.2f}%")
print(f"⏳ LSTM Sequence Model Accuracy   : {lstm_acc * 100:.2f}%")
print("========================================================")

# # =====================================================================
# # STEP 6: GENERATE CONFUSION MATRICES FOR THE DEEPER NEURAL NETWORKS
# # =====================================================================

print("\n📊 Generating 3-Class Deep Learning Confusion Matrices...")

# Get original text labels back for axis labels
class_labels = encoder.classes_

# --- 1. MLP Predictions ---
mlp_probs = mlp.predict(X_test_flat, verbose=0)
mlp_preds = np.argmax(mlp_probs, axis=1)
y_test_true = np.argmax(y_test, axis=1)

cm_mlp = confusion_matrix(y_test_true, mlp_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('MLP Confusion Matrix (Raw Data)')
plt.xlabel('Predicted Task')
plt.ylabel('True Task')
plt.tight_layout()
plt.savefig('confusion_matrix_RAW_MLP.png', dpi=300)
plt.close()

# --- 2. 2D CNN Predictions ---
cnn_probs = cnn.predict(X_test_scaled, verbose=0)
cnn_preds = np.argmax(cnn_probs, axis=1)
y_test_true = np.argmax(y_test, axis=1)

cm_cnn = confusion_matrix(y_test_true, cnn_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('2D CNN Confusion Matrix (Raw Data)')
plt.xlabel('Predicted Task')
plt.ylabel('True Task')
plt.tight_layout()
plt.savefig('confusion_matrix_RAW_CNN.png', dpi=300)
plt.close()

# --- 3. LSTM Predictions ---
lstm_probs = lstm.predict(X_test_scaled, verbose=0)
lstm_preds = np.argmax(lstm_probs, axis=1)

cm_lstm = confusion_matrix(y_test_true, lstm_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('LSTM Confusion Matrix (Raw Data)')
plt.xlabel('Predicted Task')
plt.ylabel('True Task')
plt.tight_layout()
plt.savefig('confusion_matrix_RAW_LSTM.png', dpi=300)
plt.close()
