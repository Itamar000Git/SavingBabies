import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. הגדרת נתיבים
# ==========================================
csv_dir = "csv_output"  # התיקייה שבה נמצאים כל קבצי ה-CSV הנפרדים של ההקלטות
ph_file_path = "dataset/ph_levels.csv"  # הנתיב לקובץ התשובות (רמות ה-pH)

# ==========================================
# 2. חילוץ מאפיינים מכל קבצי ה-CSV
# ==========================================
print("Extracting features from individual CSV files...")

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
features_list = []

for file in csv_files:
    # חילוץ שם הרשומה מהקובץ (למשל מ-"csv_output/1060.csv" נקבל "1060")
    record_id = os.path.splitext(os.path.basename(file))[0]

    try:
        # קריאת הקובץ
        df = pd.read_csv(file)

        # התעלמות מערכי 0 בדופק (שמסמלים לרוב נתק במוניטור)
        df['FHR'] = df['FHR'].replace(0, np.nan)

        # חישוב מאפיינים סטטיסטיים להקלטה הספציפית הזו
        fhr_mean = df['FHR'].mean()
        fhr_std = df['FHR'].std()
        fhr_min = df['FHR'].min()
        fhr_max = df['FHR'].max()

        uc_mean = df['UC'].mean()
        uc_std = df['UC'].std()
        uc_max = df['UC'].max()

        # שמירת המאפיינים במילון
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

# המרת הרשימה ל-DataFrame (טבלה)
features_df = pd.DataFrame(features_list)

# ==========================================
# 3. טעינת התשובות (pH) ואיחוד הטבלאות
# ==========================================
print("Loading pH levels and merging...")
ph_df = pd.read_csv(ph_file_path)

# מוודאים שהעמודה record_id היא מאותו סוג טקסט (String) בשתי הטבלאות
features_df['record_id'] = features_df['record_id'].astype(str)
ph_df['record_id'] = ph_df['record_id'].astype(str)

# חיבור הטבלאות - ישארו רק רשומות שיש להן גם קובץ וגם רמת pH
merged_df = pd.merge(features_df, ph_df, on='record_id', how='inner')

# ==========================================
# 4. הכנת הנתונים למודל הרגרסיה
# ==========================================
# משתנים מסבירים (X) - כל העמודות חוץ ממזהה ההקלטה ורמת ה-pH
X = merged_df.drop(columns=['record_id', 'pH'])

# משתנה המטרה (y) - רמת ה-pH שאנחנו רוצים לנבא
y = merged_df['pH']

# פיצול הנתונים: 80% לאימון המודל ו-20% לבדיקת הביצועים שלו
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. אימון המודל
# ==========================================
print("Training Linear Regression model...")
model = LinearRegression()
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

print("\n--- Feature Importance (Coefficients) ---")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.4f}")