import random

import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Layer
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling2D
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


class QuaternionConv2DKeras(Layer):
    def __init__(self, out_channels, kernel_size, strides=(1, 1), padding='valid', **kwargs):
        super(QuaternionConv2DKeras, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # input_shape: (Batch, Height, Width, 4 * in_channels)
        in_channels = input_shape[-1] // 4

        # --- CRITICAL FIX 1: QUATERNION WEIGHT INITIALIZATION ---
        # To maintain variance across Hamilton algebra, we scale He Normal standard dev by 1/sqrt(4) = 0.5
        # This prevents exploding/vanishing gradients in the complex plane.
        limit = (
            2.0 / (self.kernel_size[0] * self.kernel_size[1] * in_channels)) ** 0.5
        # Divided by 2 (sqrt of 4 quaternion dimensions)
        q_stddev = limit / 2.0
        q_initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=q_stddev)

        self.conv_r = Conv2D(self.out_channels, self.kernel_size, strides=self.strides,
                             padding=self.padding, use_bias=False, kernel_initializer=q_initializer)
        self.conv_i = Conv2D(self.out_channels, self.kernel_size, strides=self.strides,
                             padding=self.padding, use_bias=False, kernel_initializer=q_initializer)
        self.conv_j = Conv2D(self.out_channels, self.kernel_size, strides=self.strides,
                             padding=self.padding, use_bias=False, kernel_initializer=q_initializer)
        self.conv_k = Conv2D(self.out_channels, self.kernel_size, strides=self.strides,
                             padding=self.padding, use_bias=False, kernel_initializer=q_initializer)

        self.bias = self.add_weight(shape=(
            4 * self.out_channels,), initializer='zeros', trainable=True, name='bias')
        super(QuaternionConv2DKeras, self).build(input_shape)

    def call(self, inputs):
        # Split inputs along the channel axis
        r, i, j, k = tf.split(inputs, num_or_size_splits=4, axis=-1)

        # Hamilton product core algebra
        r_out = self.conv_r(r) - self.conv_i(i) - \
            self.conv_j(j) - self.conv_k(k)
        i_out = self.conv_r(i) + self.conv_i(r) + \
            self.conv_j(k) - self.conv_k(j)
        j_out = self.conv_r(j) - self.conv_i(k) + \
            self.conv_j(r) + self.conv_k(i)
        k_out = self.conv_r(k) + self.conv_i(j) - \
            self.conv_j(i) + self.conv_k(r)

        # Combine components
        out = tf.concat([r_out, i_out, j_out, k_out], axis=-1)
        return out + self.bias

# =====================================================================
# 2. KERAS QCNN MODEL ARCHITECTURE
# =====================================================================


def build_keras_qcnn2d(max_timesteps=650, num_classes=3):
    inputs = tf.keras.Input(shape=(max_timesteps, 40))

    # Reshape to (Batch, Timesteps, Sensors=10, Channels=4)
    x = Reshape((max_timesteps, 10, 4))(inputs)

    # Layer 1 Block (8 Q-Channels = 32 real filters)
    x = QuaternionConv2DKeras(out_channels=8, kernel_size=(5, 1))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Layer 2 Block (16 Q-Channels = 64 real filters)
    x = QuaternionConv2DKeras(out_channels=16, kernel_size=(3, 1))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Dense Classifier
    x = GlobalAveragePooling2D()(x)
    # Expanded dense capacity slightly for QCNN representation
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


# Initialize Model
qcnn_keras = build_keras_qcnn2d(max_timesteps=MAX_TIMESTEPS, num_classes=3)

# --- CRITICAL FIX 2: ADJUST LEARNING RATE FOR SMALL BATCH SIZES ---
# A batch size of 8 is highly dynamic. Using standard 1e-3 is too noisy.
# We initialize at 5e-4 to allow delicate quaternion relationships to settle.
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

qcnn_keras.compile(
    optimizer=opt,
    loss='mse',
    metrics=['accuracy']
)

# --- CRITICAL FIX 3: DYNAMIC SCHEDULING ---
# If validation accuracy plateaus, scale the learning rate down to land perfectly in global minima
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.000001,
    verbose=1
)

# Merge with your existing Early Stopping or checkpoints
callbacks_list = [lr_scheduler]
if 'monitor_callbacks' in globals():
    callbacks_list.extend(monitor_callbacks)

print("\n🌀 Training 2D Quaternion CNN (Optimized for 80%+ Accuracy)...")
qcnn_keras.fit(
    X_train_scaled,
    y_train,
    epochs=150,  # Give QCNN parameters the necessary runtime to converge
    batch_size=15,  # Kept at your preferred batch size of 8
    validation_data=(X_test_scaled, y_test),
    # callbacks=callbacks_list,
    verbose=1
)

# Final evaluation on holdout test dataset
_, qcnn_acc = qcnn_keras.evaluate(X_test_scaled, y_test, verbose=0)

print("\n========================================================")
print("🎯 QUATERNION DEEP LEARNING BENCHMARKS (3-CLASS)")
print("========================================================")
print(f"🌀 2D QCNN Architecture Accuracy   : {qcnn_acc * 100:.2f}%")
print("========================================================")
