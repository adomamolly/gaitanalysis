import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

# =====================================================================
# 1. PARAMETERS & CONFIGURATION
# =====================================================================
LEG_LENGTH = 0.85    # l (meters) for Stride Length math
TOTAL_DIST = 7.33    # d (meters) for Gait Speed math

# Target segment for primary gait equations
target_segment = 'rt-ua'
quat_cols = [f'{target_segment}x', f'{target_segment}y',
             f'{target_segment}z', f'{target_segment}w']

# Master list to accumulate spreadsheet rows
compiled_results = []

# =====================================================================
# 2. SEQUENTIAL FOLDER LOOP (N01 to N47)
# =====================================================================
n = 1
while n < 48:
    subject_str = f"N{n:02d}"
    folder_path = os.path.join('sd', f'SE1-{subject_str}')

    print("=" * 75)
    print(f"📁 PROCESSING SUBJECT FOLDER: {folder_path}")
    print("=" * 75)

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder {folder_path} not found. Skipping.")
        n += 1
        continue

    file_pattern = os.path.join(folder_path, "*.txt")
    subject_files = glob.glob(file_pattern)

    # Dictionary to store pitch profiles for this subject's final single comparison plot
    subject_plots_data = {}

    # --- Loop through subfiles (HT, LS, OB, TD) ---
    for file_path in sorted(subject_files):
        file_name = os.path.basename(file_path)
        task_type = os.path.splitext(file_name)[0].split('-')[-1]

        try:
            df = pd.read_csv(file_path, sep=r'\s+')
        except Exception as e:
            print(f"   ❌ Read Error on {file_name}: {e}")
            continue

        if not all(col in df.columns for col in quat_cols):
            continue

        # --- Step A: Missing Data Check & Imputation ---
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            df = df.interpolate(method='linear').bfill()

        # --- NEW REQUIREMENT: Identify Inactive Sensors via Variance ---
        # Find columns where variance is near-zero (dead/inactive sensors)
        variances = df.var()
        inactive_cols = variances[variances < 0.001].index.tolist()
        inactive_sensors_string = ", ".join(
            inactive_cols) if inactive_cols else "None"

        # --- NEW REQUIREMENT: Calculate Time by Dividing Rows by 59 ---
        total_rows = len(df)
        total_duration = total_rows / 59.0  # Dynamic time allocation
        sampling_rate_calculated = 59.0     # Effective Hz

        # --- Step B: Quaternion to Euler Conversion ---
        q_right = df[['rt-uax', 'rt-uay', 'rt-uaz', 'rt-uaw']].to_numpy()
        pitch_r = R.from_quat(q_right / np.linalg.norm(q_right,
                              axis=1, keepdims=True)).as_euler('xyz', degrees=True)[:, 1]

        # Store pitch signal for the final overlay visualization matrix
        subject_plots_data[task_type] = pitch_r

        # --- Step C: Peak Detection & Feature Extraction ---
        # Lowered distance threshold for 59Hz scaling
        peaks_r, _ = find_peaks(pitch_r, distance=25, prominence=5)
        valleys_r, _ = find_peaks(-pitch_r, distance=25, prominence=5)

        H_time_r = (valleys_r / sampling_rate_calculated)
        T_time_r = (peaks_r / sampling_rate_calculated)
        num_cycles = min(len(H_time_r) - 1, len(T_time_r))

        if num_cycles > 1:
            gait_speed = (TOTAL_DIST / total_duration) * 60
            cadence = (len(peaks_r) / 2) / (total_duration / 60)

            double_support_sum = 0
            stride_lengths = []
            swing_phases = []

            for i in range(num_cycles):
                double_support_sum += (T_time_r[i] - H_time_r[i])
                angle_i1 = np.radians(pitch_r[valleys_r[i+1]])
                angle_i = np.radians(pitch_r[valleys_r[i]])
                stride_lengths.append(
                    LEG_LENGTH * np.sin(angle_i1) + LEG_LENGTH * np.sin(angle_i))
                swing_phases.append(H_time_r[i+1] - T_time_r[i])

            avg_stride_len = np.abs(np.mean(stride_lengths)) * 100
            double_support = double_support_sum / total_duration
            avg_swing = np.mean(swing_phases)

            # Save metrics AND inactive sensor classifications to list
            compiled_results.append({
                'Subject': subject_str,
                'Task': task_type,
                'Total_Rows': total_rows,
                'Calculated_Duration_sec': round(total_duration, 2),
                'Stride_Length_cm': round(avg_stride_len, 2),
                'Gait_Speed_m_min': round(gait_speed, 2),
                'Cadence_steps_min': round(cadence, 2),
                'Double_Support_Ratio': round(double_support, 3),
                'Swing_Phase_sec': round(avg_swing, 3),
                'Inactive_Sensors_Count': len(inactive_cols),
                'Inactive_Sensors_List': inactive_sensors_string
            })
            print(
                f"   ✅ Processed {task_type}: {total_rows} rows ➔ {total_duration:.2f}s. Inactive found: {len(inactive_cols)}")

    # =====================================================================
    # 3. NEW REQUIREMENT: VISUALIZE EACH GAIT TYPE IN A SINGLE OVALY PLOT
    # =====================================================================
    if subject_plots_data:
        plt.figure(figsize=(11, 5))

        # Layer each available gait task onto one single canvas context
        for task_suffix, pitch_signal in subject_plots_data.items():
            # Plot up to 300 frames so variations are easy to see close up
            plt.plot(
                pitch_signal[:300], label=f"Task: {task_suffix}", linewidth=2, alpha=0.8)

        plt.title(
            f"Gait Type Comparison Matrix - Subject {subject_str}", fontsize=12, fontweight='bold')
        plt.xlabel("Normalized Time Frames (59 Hz Local Scale)")
        plt.ylabel("Upper Arm Pitch Angle (Degrees)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')

        os.makedirs('gait_comparisons', exist_ok=True)
        plt.savefig(
            f'gait_comparisons/{subject_str}_gait_comparison.png', bbox_inches='tight')
        plt.close()
        print(
            f"   📊 Saved single multi-gait visualization chart for {subject_str}.")

    n += 1  # Move to next folder numbers

# =====================================================================
# 4. EXPORT MASTER EXCEL DATABASE
# =====================================================================
print("\n" + "=" * 75)
print("📊 COMPILING FINAL CLASSIFIED EXCEL DATASET")
print("=" * 75)

summary_df = pd.DataFrame(compiled_results)
excel_output_path = 'Gait_Analysis_Master_Report.xlsx'
summary_df.to_excel(excel_output_path, index=False,
                    sheet_name='Gait Parameters')

print(f"✨ Success! Summary compiled for {len(summary_df)} active subfiles.")
print(
    f"📁 Spreadsheet updated with inactive sensors & 59Hz duration math: '{excel_output_path}'")
