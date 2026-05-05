import pandas as pd
import os
import glob

def add_be_and_risk(csv_path, hea_dir, output_csv_path):
    # 1. קריאת קובץ ה-CSV הקיים
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_path}")
        return
        
    # נוודא שעמודת המזהה היא מסוג מחרוזת (String) כדי שנוכל לחבר נתונים
    id_col = 'record_id' 
    if id_col not in df.columns:
        print(f"Error: Column '{id_col}' not found in CSV. Please update 'id_col' variable.")
        return
        
    df[id_col] = df[id_col].astype(str)
    
    # 2. חילוץ נתוני ה-BE מקבצי ה-.hea
    be_data = {}
    
    # נניח שקבצי ה-.hea נמצאים בתיקיית המקור (או בתיקייה שתגדיר)
    hea_files = glob.glob(os.path.join(hea_dir, "*.hea"))
    print(f"Found {len(hea_files)} .hea files in {hea_dir}...")
    
    for file in hea_files:
        record_id = os.path.basename(file).split(".")[0]
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('#BDecf'):
                    # חילוץ המספר מהשורה, למשל מתוך "#BE           -10.5"
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # לוקחים את האיבר האחרון בשורה שיהיה המספר
                            be_data[record_id] = float(parts[-1])
                        except ValueError:
                            pass
                    break # מצאנו את ה-BE, אפשר לעבור לקובץ הבא
    
    # 3. הוספת ה-BE הגולמי ל-DataFrame
    df['BE'] = df[id_col].map(be_data).abs()
    
    # 4. הוספת עמודה של הערך המוחלט של BE
    # פונקציית abs() הופכת לדוגמה מינוס 10.5 ל-10.5 פלוס
    
    # 5. יצירת עמודת תווית הסיכון: 1 אם הערך המוחלט גדול מ-8, אחרת 0
    df['riskBE'] = (df['BE'] > 8.0).astype(int)
    
    # 6. שמירת הקובץ המעודכן
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nSuccessfully saved updated CSV to: {output_csv_path}")
    print("\nPreview of the updated data:")
    print(df[[id_col, 'BE', 'riskBE']].head(10))
    print(f"\nClass counts for riskBE (Normal=0, Danger=1):")
    print(df['riskBE'].value_counts())

# --- הפעלת הסקריפט ---
if __name__ == "__main__":
    # הנתיב לקובץ ה-CSV שעליו אתה רוצה להוסיף את העמודות
    input_csv = "ph_levels_with_risk.csv" 
    
    # התיקייה בה נמצאים קבצי ה-.hea המקוריים (ולא ה-info שיצרנו קודם)
    # שנה את הנתיב הזה לתיקייה שבה שמורים כל קבצי ה-hea שלך!
    hea_directory = "." 
    
    # שם הקובץ החדש שיווצר
    output_csv = "ph_levels_with_be_risk.csv"
    
    add_be_and_risk(input_csv, hea_directory, output_csv)