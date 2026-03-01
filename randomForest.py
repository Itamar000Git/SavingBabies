import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# הצגת חשיבות המאפיינים (Feature Importances)
print("\n--- Feature Importance ---")
# יצירת טבלה שמסדרת את המאפיינים לפי רמת החשיבות שלהם (מהכי משפיע להכי פחות)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

for index, row in feature_importance_df.iterrows():
    # נציג את החשיבות כאחוזים כדי שיהיה קל יותר להבין
    print(f"{row['Feature']}: {row['Importance'] * 100:.2f}%")