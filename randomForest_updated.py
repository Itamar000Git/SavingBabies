import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ==========================================
# 1. Paths
# ==========================================
csv_dir = "csv_output"
ph_file_path = "dataset/ph_levels.csv"

# ==========================================
# 2. Feature extraction config (LAST WINDOW)
# ==========================================
WINDOW_MINUTES = 30
FS = 4
WINDOW_SAMPLES = WINDOW_MINUTES * 60 * FS

print(f"Extracting features from LAST {WINDOW_MINUTES} minutes (={WINDOW_SAMPLES} samples)...")

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
features_list = []

def safe_stats(series: pd.Series):
    """Return mean/std/min/max for a pandas Series ignoring NaNs."""
    return (
        series.mean(skipna=True),
        series.std(skipna=True),
        series.min(skipna=True),
        series.max(skipna=True),
    )

for file in csv_files:
    record_id = os.path.splitext(os.path.basename(file))[0]

    try:
        df = pd.read_csv(file)

        # Basic validation
        if "FHR" not in df.columns or "UC" not in df.columns:
            print(f"Skipping {record_id}: missing FHR/UC columns")
            continue

        # Take last window
        df_last = df.iloc[-WINDOW_SAMPLES:] if len(df) >= WINDOW_SAMPLES else df.copy()

        # Replace 0 (monitor dropout) with NaN for FHR
        df_last["FHR"] = df_last["FHR"].replace(0, np.nan)

        # Quality feature: valid ratio of FHR in the window
        fhr_valid_ratio = df_last["FHR"].notna().mean()

        fhr_mean, fhr_std, fhr_min, fhr_max = safe_stats(df_last["FHR"])
        uc_mean, uc_std, uc_min, uc_max = safe_stats(df_last["UC"])

        features_list.append({
            "record_id": str(record_id),
            "window_minutes": WINDOW_MINUTES,
            "window_samples_used": int(len(df_last)),

            # data quality
            "FHR_valid_ratio": float(fhr_valid_ratio),

            # FHR stats (last window)
            "FHR_mean_last": 0 if np.isnan(fhr_mean) else float(fhr_mean),
            "FHR_std_last":  0 if np.isnan(fhr_std)  else float(fhr_std),
            "FHR_min_last":  0 if np.isnan(fhr_min)  else float(fhr_min),
            "FHR_max_last":  0 if np.isnan(fhr_max)  else float(fhr_max),

            # UC stats (last window)
            "UC_mean_last": 0 if np.isnan(uc_mean) else float(uc_mean),
            "UC_std_last":  0 if np.isnan(uc_std)  else float(uc_std),
            "UC_min_last":  0 if np.isnan(uc_min)  else float(uc_min),
            "UC_max_last":  0 if np.isnan(uc_max)  else float(uc_max),
        })

    except Exception as e:
        print(f"Error processing {file}: {e}")

features_df = pd.DataFrame(features_list)

# ==========================================
# 3. Load pH labels and merge
# ==========================================
print("Loading pH levels and merging...")
ph_df = pd.read_csv(ph_file_path)
ph_df["record_id"] = ph_df["record_id"].astype(str)

merged_df = pd.merge(features_df, ph_df, on="record_id", how="inner")

# Filter out very low-quality windows
MIN_VALID_RATIO = 0.70
before = len(merged_df)
merged_df = merged_df[merged_df["FHR_valid_ratio"] >= MIN_VALID_RATIO].copy()
after = len(merged_df)
print(f"Filtered by FHR_valid_ratio >= {MIN_VALID_RATIO}: {before} -> {after}")

# ==========================================
# 4. Prepare X/y
# ==========================================
X = merged_df.drop(columns=["record_id", "pH"])
y = merged_df["pH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 5. Train Random Forest Regressor
# ==========================================
print("Training Random Forest Regressor model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 6. Evaluate regression
# ==========================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Evaluation ---")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R^2:  {r2:.4f}")

# ==========================================
# 7. Feature importances
# ==========================================
print("\n--- Feature Importance ---")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

for _, row in feature_importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance'] * 100:.2f}%")

# ==========================================
# 8. Classification metrics (Accuracy/Precision/Recall/F1)
#    Derived from predicted pH using a threshold
# ==========================================
PH_THRESHOLD = 7.05  # change to 7.10 / 7.15 if you want a less strict definition

y_test_cls = (y_test <= PH_THRESHOLD).astype(int)   # 1 = acidemia
y_pred_cls = (y_pred <= PH_THRESHOLD).astype(int)

acc = accuracy_score(y_test_cls, y_pred_cls)
pre = precision_score(y_test_cls, y_pred_cls, zero_division=0)
rec = recall_score(y_test_cls, y_pred_cls, zero_division=0)
f1  = f1_score(y_test_cls, y_pred_cls, zero_division=0)
cm  = confusion_matrix(y_test_cls, y_pred_cls)

print("\n--- Classification (derived from pH) ---")
print(f"Pathological if pH <= {PH_THRESHOLD}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:")
print(cm)