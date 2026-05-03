import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

# ==========================================
# 1. הגדרת נתיבים
# ==========================================
csv_dir = "csv_output"  # התיקייה שבה נמצאים כל קבצי ה-CSV של ההקלטות
ph_file_path = "dataset/ph_levels.csv"  # הנתיב לקובץ התשובות (רמות ה-pH)

# ==========================================
# 2. חילוץ מאפיינים מכל קבצי ה-CSV
# ==========================================
print("Extracting features from individual CSV files...")

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
features_list = []

for file in csv_files:
    record_id = os.path.splitext(os.path.basename(file))[0]

    try:
        df = pd.read_csv(file)

        # התעלמות מערכי 0 בדופק (שמסמלים לרוב נתק במוניטור)
        df['FHR'] = df['FHR'].replace(0, np.nan)

        # חישוב מאפיינים
        fhr_mean = df['FHR'].mean()
        fhr_std = df['FHR'].std()
        fhr_min = df['FHR'].min()
        fhr_max = df['FHR'].max()

        uc_mean = df['UC'].mean()
        uc_std = df['UC'].std()
        uc_max = df['UC'].max()

        features_list.append({
            'record_id': record_id,
            'FHR_mean': fhr_mean if not np.isnan(fhr_mean) else 0,
            'FHR_std': fhr_std if not np.isnan(fhr_std) else 0,
            'FHR_min': fhr_min if not np.isnan(fhr_min) else 0,
            'FHR_max': fhr_max if not np.isnan(fhr_max) else 0,
            'UC_mean': uc_mean if not np.isnan(uc_mean) else 0,
            'UC_std': uc_std if not np.isnan(uc_std) else 0,
            'UC_max': uc_max if not np.isnan(uc_max) else 0
        })

    except Exception as e:
        print(f"Error processing {file}: {e}")

features_df = pd.DataFrame(features_list)

# ==========================================
# 3. טעינת התשובות (pH) ואיחוד הטבלאות
# ==========================================
print("Loading pH levels and merging...")
ph_df = pd.read_csv(ph_file_path)

features_df['record_id'] = features_df['record_id'].astype(str)
ph_df['record_id'] = ph_df['record_id'].astype(str)

merged_df = pd.merge(features_df, ph_df, on='record_id', how='inner')

# ==========================================
# 4. הכנת הנתונים למודל
# ==========================================
X = merged_df.drop(columns=['record_id', 'pH'])
y = merged_df['pH']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. אימון מודל Random Forest
# ==========================================
print("Training Random Forest Regressor model...")
# n_estimators=100 אומר למודל ליצור 100 עצי החלטה שונים ולאחד את התוצאות שלהם
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 6. בדיקת ביצועי המודל
# ==========================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ==========================================
# 7. Classification metrics (Accuracy/Precision/Recall/F1)
#    ע"י הגדרת חמצת לפי סף pH
# ==========================================
PH_THRESHOLD = 7.05  # אפשר לשנות (למשל 7.10 / 7.15)

# אמת: האם יש חמצת
y_test_cls = (y_test <= PH_THRESHOLD).astype(int)

# תחזית: האם יש חמצת לפי תחזית pH
y_pred_cls = (y_pred <= PH_THRESHOLD).astype(int)

acc = accuracy_score(y_test_cls, y_pred_cls)
pre = precision_score(y_test_cls, y_pred_cls, zero_division=0)
rec = recall_score(y_test_cls, y_pred_cls, zero_division=0)
f1  = f1_score(y_test_cls, y_pred_cls, zero_division=0)
cm  = confusion_matrix(y_test_cls, y_pred_cls)

print("\n--- Classification (derived from pH) ---")
print(f"Threshold (pathological if pH <= {PH_THRESHOLD})")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:")
print(cm)

# ==========================================
# 8. Feature importances
# ==========================================
print("\n--- Feature Importance ---")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

for _, row in feature_importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance'] * 100:.2f}%")