import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks, resample

# =====================================================================
# 1. PARAMETERS & CONFIGURATION
# =====================================================================
LEG_LENGTH = 0.85    # l (meters) for Stride Length math
TOTAL_DIST = 7.33    # d (meters) for Gait Speed math

# Target segment for primary gait equations
target_segment = 'rt-ll'
quat_cols = [f'{target_segment}x', f'{target_segment}y',
             f'{target_segment}z', f'{target_segment}w']

# List of 10 actual physical sensor modules to track integrity
body_segments = [
    'rt-ua', 'rt-fa', 'rt-th', 'rt-ll', 'rt-ft',
    'lt-ua', 'lt-fa', 'lt-th', 'lt-ll', 'lt-ft'
]

# Master list to accumulate spreadsheet rows
compiled_results = []

# Dictionary to collect resampled pitch signals globally across all subjects for Trend Plots
STANDARD_LENGTH = 500
global_trends = {'LS': [], 'HT': [], 'OB': [], 'TD': []}
gait_labels = {'LS': 'Level Surf',
               'HT': 'H Turns', 'OB': 'Obstacle', 'TD': 'Tandem'}

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

    # --- Loop through subfiles (HT, LS, OB, TD) ---
    for file_path in sorted(subject_files):
        file_name = os.path.basename(file_path)
        task_type = os.path.splitext(file_name)[0].split('-')[-1].upper()

        try:
            df = pd.read_csv(file_path, sep=r'\s+')
        except Exception as e:
            print(f"   ❌ Read Error on {file_name}: {e}")
            continue

        if not all(col in df.columns for col in quat_cols):
            print(f"   ⚠️ Skipping: Missing sensor columns in {file_name}")
            continue

        # --- Step A: Missing Data Check & Imputation ---
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            df = df.interpolate(method='linear').bfill()

        # --- UPDATE: Identify Inactive Sensors by Whole Body Segment Modules ---
        variances = df.var()
        inactive_segments = []

        for seg in body_segments:
            segment_cols = [f'{seg}x', f'{seg}y', f'{seg}z', f'{seg}w']
            seg_vars = [variances.get(col, 0) for col in segment_cols]

            # If ALL 4 quaternion axes for a segment have variance < 0.001, it's inactive
            if all(v < 0.001 for v in seg_vars):
                inactive_segments.append(seg)

        inactive_sensors_string = ", ".join(
            inactive_segments) if inactive_segments else "None"

        # --- Step B: Calculate Time by Dividing Rows by 59 ---
        total_rows = len(df)
        total_duration = total_rows / 59.0
        sampling_rate_calculated = 59.0

        # --- Step C: Quaternion to Euler Conversion ---
        q_right = df[['rt-llx', 'rt-lly', 'rt-llz', 'rt-llw']].to_numpy()
        pitch_r = R.from_quat(q_right / np.linalg.norm(q_right,
                              axis=1, keepdims=True)).as_euler('xyz', degrees=True)[:, 1]

        # Append resampled clean data to global repository for the master trend visualization
        if task_type in global_trends:
            global_trends[task_type].append(resample(pitch_r, STANDARD_LENGTH))

        # --- Step D: Peak Detection & Feature Extraction ---
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

            # Save metrics AND module-level inactive sensor data to master list
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
                'Inactive_Sensors_Count': len(inactive_segments),
                'Inactive_Sensors_List': inactive_sensors_string
            })
            print(
                f"   ✅ Processed {task_type}: {total_rows} rows ➔ {total_duration:.2f}s. Inactive Modules: {len(inactive_segments)}")

    n += 1

# =====================================================================
# 3. NEW REQUIREMENT: MASTER GLOBAL COHORT TREND OVERLAY PLOT
# =====================================================================
print("\n" + "=" * 75)
print("📊 GENERATING UNIFIED COHORT TREND GRAPH")
print("=" * 75)

plt.figure(figsize=(13, 6.5))
colors = {'LS': '#1f77b4', 'HT': '#ff7f0e', 'OB': '#2ca02c', 'TD': '#d62728'}
time_percent = np.linspace(0, 100, STANDARD_LENGTH)

for task_suffix, label in gait_labels.items():
    data_matrix = np.array(global_trends[task_suffix])

    if len(data_matrix) == 0:
        continue

    # Calculate group mean trend and standard deviation ribbon
    mean_waveform = np.mean(data_matrix, axis=0)
    std_waveform = np.std(data_matrix, axis=0)

    # Plot solid mean path line
    plt.plot(time_percent, mean_waveform,
             label=f'{label} (Trend, n={len(data_matrix)})', color=colors[task_suffix], linewidth=2.5)
    # Fill standard deviation uncertainty band
    plt.fill_between(time_percent, mean_waveform - std_waveform,
                     mean_waveform + std_waveform, color=colors[task_suffix], alpha=0.15)

plt.title("Universal Gait Type Fingerprints: Cohort Population Trends (N01 - N47)",
          fontsize=13, fontweight='bold')
plt.xlabel("Gait Trial Progress Timeline (%)", fontsize=11)
plt.ylabel("Lower Leg Pitch Angle (Degrees)", fontsize=11)
plt.xlim(0, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
           fontsize=10, borderaxespad=0)
plt.subplots_adjust(right=0.80)

plt.savefig('Global_Gait_Type_Trends.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Consolidated 'Global_Gait_Type_Trends.png' exported successfully.")

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
print(f"📁 Master Spreadsheet saved: '{excel_output_path}'")
