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
    ['LS', 'OB'],
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

    # =====================================================================
    # 1. RANDOM FOREST TRAINING & CONFUSION MATRIX GENERATION
    # =====================================================================
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    # Get predictions
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    # Generate Confusion Matrix for this specific pair
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm_rf = confusion_matrix(y_test, rf_preds)

    # Set up the plot aesthetics
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt='d',
        cmap='Greens',  # Forest green theme to easily distinguish it from the SVM plots
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    # Add clear slide titles and labels
    plt.title(f'Random Forest Confusion Matrix: {pair[0]} vs {pair[1]}')
    plt.xlabel('Predicted Task')
    plt.ylabel('True Task')
    plt.tight_layout()

    # Save the file automatically with a unique name based on the pair
    filename_rf = f"confusion_matrix_RF_{pair[0]}_vs_{pair[1]}.png"
    plt.savefig(filename_rf, dpi=300)
    plt.close()  # Clear memory

    print(f"🌲 Saved Random Forest graphic as: '{filename_rf}'")

    # Store results for this pair
    all_pair_results.append({
        'Comparison Pair': f"{pair[0]} vs {pair[1]}",
        'Random Forest': rf_acc,
    })

    # =====================================================================
    # 2. SVM TRAINING & CONFUSION MATRIX GENERATION
    # =====================================================================
    svm = SVC(kernel='rbf', C=5.0, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Get predictions
    svm_preds = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_preds)

    # Generate Confusion Matrix for this specific pair
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, svm_preds)

    # Set up the plot aesthetics
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Purples',  # Sleek purple theme to stand out in your slides
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    # Add clear slide titles and labels
    plt.title(f'SVM Confusion Matrix: {pair[0]} vs {pair[1]}')
    plt.xlabel('Predicted Task')
    plt.ylabel('True Task')
    plt.tight_layout()

    # Save the file automatically with a unique name based on the pair
    filename = f"confusion_matrix_{pair[0]}_vs_{pair[1]}.png"
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the plot to clear system memory for the next loop run

    print(f"🖼️ Saved confusion matrix graphic as: '{filename}'")

# Convert to DataFrame and show final table
master_results_df = pd.DataFrame(all_pair_results)
print("\n========================================================")
print("🎯 MASTER BINARY RESULTS FOR YOUR SLIDES")
print("========================================================")
print(master_results_df.to_string(index=False))

# Save directly to your repository folder
master_results_df.to_csv('gait_binary_master_results.csv', index=False)
print("\n💾 Saved master spreadsheet as: 'gait_binary_master_results.csv'")
