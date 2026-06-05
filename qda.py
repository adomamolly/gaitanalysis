import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

# =====================================================================
# 1. SETUP PARAMETERS & CONSTANTS
# =====================================================================
file_path = 'sd/SE1-N25/SE1-N25-HT.txt'
sampling_rate = 100  # 100 Hz sampling frequency
leg_length = 0.85    # Leg length (l) in meters for Stride Length math
total_dist = 7.33    # Distance traveled (d) in meters for Gait Speed math

# =====================================================================
# 2. LOAD DATASET
# =====================================================================
# Load data handling whitespace separation
df = pd.read_csv(file_path, sep=r'\s+')

# =====================================================================
# 3. MISSING DATA CHECK & LINEAR INTERPOLATION (IMPUTATION)
# =====================================================================
# Count total missing (NaN) values in the file
total_missing = df.isnull().sum().sum()

if total_missing > 0:
    print(
        f"⚠️ Found {total_missing} missing values in {file_path}. Patching data...")
    # Use linear interpolation to fill gaps, and bfill to handle any NaNs at row 0
    df = df.interpolate(method='linear').bfill()
    print("   ✅ Imputation complete. Data is now continuous.")
else:
    print("✨ Data check passed: No missing data found.")

# =====================================================================
# 4. EXTRACT RELEVANT QUATERNIONS
# =====================================================================
# Extract right lower leg (rt-ll) and left lower leg (lt-ll) columns (x, y, z, w)
q_right = df[['rt-llx', 'rt-lly', 'rt-llz', 'rt-llw']].to_numpy()
q_left = df[['lt-llx', 'lt-lly', 'lt-llz', 'lt-llw']].to_numpy()

# =====================================================================
# 5. QUATERNION TO EULER CONVERSION (PITCH ANGLE)
# =====================================================================
# Convert normalized quaternions to Euler angles and isolate the Pitch (Y axis)
pitch_r = R.from_quat(q_right / np.linalg.norm(q_right,
                      axis=1, keepdims=True)).as_euler('xyz', degrees=True)[:, 1]
pitch_l = R.from_quat(q_left / np.linalg.norm(q_left, axis=1,
                      keepdims=True)).as_euler('xyz', degrees=True)[:, 1]

# Create timestamps array based on the length of the dataframe
timestamps = np.arange(len(df)) / sampling_rate
total_duration = timestamps[-1]  # T_total

# =====================================================================
# 6. GAIT EVENT DETECTION (HEEL STRIKES & TOE OFFS)
# =====================================================================
# Identify peaks (Mid-Swings) and valleys (Heel Strikes / Initial Contact 'H')
peaks_r, _ = find_peaks(pitch_r, distance=40, prominence=5)
valleys_r, _ = find_peaks(-pitch_r, distance=40, prominence=5)
valleys_l, _ = find_peaks(-pitch_l, distance=40, prominence=5)

# Vector H: Times of initial heel contact (valleys)
H_time_r = timestamps[valleys_r]
H_time_l = timestamps[valleys_l]

# Vector T: Times of opposite toe-off (peaks)
T_time_r = timestamps[peaks_r]

# Bound the loops by the shortest detected event array
num_cycles = min(len(H_time_r) - 1, len(T_time_r))

# =====================================================================
# 7. MATHEMATICAL EQUATIONS IMPLEMENTATION
# =====================================================================
# Equation (3): Gait Speed = d / T_total (Converted to m/min)
gait_speed_m_min = (total_dist / total_duration) * 60

# Cadence: Number of peaks divided by 2 over total minutes
cadence_steps_min = (len(peaks_r) / 2) / (total_duration / 60)

# Stride loop variables
double_support_sum = 0
stride_lengths = []
swing_phases_r = []

for i in range(num_cycles):
    # Equation (1): Double Support [ T(i) - H(i) ]
    double_support_sum += (T_time_r[i] - H_time_r[i])

    # Equation (2): Stride Length = l * sin(H(i+1)) + l * sin(H(i))
    # Note: angles must be converted to radians for numpy sine functions
    angle_i1 = np.radians(pitch_r[valleys_r[i+1]])
    angle_i = np.radians(pitch_r[valleys_r[i]])
    stride_lengths.append(leg_length * np.sin(angle_i1) +
                          leg_length * np.sin(angle_i))

    # Equation (4): Swing Phase = H(i+1) - T(i)
    swing_phases_r.append(H_time_r[i+1] - T_time_r[i])

# Final parameter average realization
double_support = double_support_sum / total_duration
avg_stride_length_cm = np.abs(np.mean(stride_lengths)) * 100  # Convert to cm
avg_swing_phase_r = np.mean(swing_phases_r)

# =====================================================================
# 8. PRINT GAIT SUMMARY REPORT
# =====================================================================
print(f"\n--- Calculated Gait Parameters for {file_path.split('/')[-1]} ---")
print(f"Stride Length   : {avg_stride_length_cm:.1f} cm")
print(f"Gait Speed      : {gait_speed_m_min:.1f} m/min")
print(f"Cadence         : {cadence_steps_min:.1f} steps/min")
print(f"Double Support  : {double_support:.2f}")
print(f"Swing Phase (R) : {avg_swing_phase_r:.2f} sec")
