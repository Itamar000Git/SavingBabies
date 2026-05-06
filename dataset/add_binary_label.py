import pandas as pd

def add_risk_column(input_csv, output_csv):
    # 1. קריאת קובץ ה-CSV לתוך DataFrame
    df = pd.read_csv(input_csv)
    
    # הנחה: שם העמודה שמכילה את הערכים בקובץ הוא 'pH'. 
    # אם השם שונה (למשל 'ph_level'), יש לשנות את השם כאן.
    ph_column_name = 'pH' 
    
    # 2. הוספת עמודת RISK
    # התנאי: אם ה-pH קטן מ-7.1 הערך יהיה 1 (יש סיכון), אחרת 0 (אין סיכון)
    # הפונקציה astype(int) הופכת את התוצאה הבוליאנית (True/False) ל-1 ו-0.
    df['RISK'] = (df[ph_column_name] < 7.1).astype(int)
    
    # אם תעדיף טקסט ('Yes' / 'No') במקום מספרים, תוכל להשתמש בשורה הזו במקום הקודמת:
    # df['RISK'] = df[ph_column_name].apply(lambda x: 'Yes' if x < 7.1 else 'No')

    # 3. שמירת הנתונים לקובץ CSV חדש כדי לא לדרוס את המקור
    df.to_csv(output_csv, index=False)
    print(f"File saved successfully to {output_csv}")
    
    # הצגת 5 השורות הראשונות כדי לוודא שזה עבד
    print(df.head())

# הרצת הקוד על הקובץ שלך
if __name__ == "__main__":
    input_file = "ph_levels.csv"
    output_file = "ph_levels
    _with_risk.csv"
    
    add_risk_column(input_file, output_file)