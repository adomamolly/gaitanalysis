import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load the master summary spreadsheet
df_ml = pd.read_excel('Gait_Analysis_Master_Report.xlsx')

# Preprocess the data

X = df_ml.drop(columns=['Subject', 'Task', 'Inactive_Sensors_List',
               'Inactive_Sensors_Count', 'Total_Rows', 'Calculated_Duration_sec'])
y = df_ml['Task']

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
