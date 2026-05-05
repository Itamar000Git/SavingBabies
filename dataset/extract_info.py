import os
import re
import json

# רשימת המאפיינים המותרים לשימוש (נתונים שידועים *לפני* הלידה בלבד)
ALLOWED_FEATURES = [
    'Gest. weeks',  # שבוע היריון
    'Sex',          # מין העובר
    'Age',          # גיל האם
    'Gravidity',    # מספר הריונות
    'Parity',       # מספר לידות בעבר
    'Diabetes',     # סוכרת
    'Hypertension', # יתר לחץ דם
    'Preeclampsia', # רעלת היריון
    'Liq. praecox', # ירידת מים מוקדמת
    'Pyrexia',      # חום
    'Meconium',     # מים מיקוניאליים
    'Presentation', # מצג העובר
    'Induced'       # האם הלידה מושרית
]

def create_clean_info_folder(source_dir=".", output_dir="info"):
    """
    הפונקציה עוברת על כל קבצי ה-hea בתיקיית המקור, מחלצת רק נתונים רלוונטיים,
    ויוצרת קובץ JSON נפרד לכל מטופל בתיקיית info.
    """
    # 1. יצירת תיקיית info אם היא עדיין לא קיימת
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created/Accessed directory: '{output_dir}'")
    
    # 2. מעבר על כל הקבצים בתיקיית המקור
    for filename in os.listdir(source_dir):
        if filename.endswith(".hea"):
            # חילוץ מספר המטופל מהשם של הקובץ (למשל '1001' מתוך '1001.hea')
            patient_id = filename.split('.')[0]
            filepath = os.path.join(source_dir, filename)
            
            patient_data = {}
            
            # 3. קריאת הקובץ וחילוץ הנתונים
            with open(filepath, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    # חיפוש שורות של נתונים (מתחילות ב-# אבל לא ב-#-)
                    if line.startswith('#') and not line.startswith('#-'):
                        content = line[1:].strip()
                        parts = re.split(r'\s{2,}|\t+', content)
                        
                        if len(parts) >= 2:
                            key = parts[0].strip()
                            value_str = parts[-1].strip()
                            
                            # המרה של המחרוזת למספר עשרוני או שלם
                            try:
                                value = float(value_str) if '.' in value_str else int(value_str)
                            except ValueError:
                                value = value_str
                            
                            # 4. סינון: שומרים את הנתון רק אם הוא ברשימה המותרת
                            if key in ALLOWED_FEATURES:
                                patient_data[key] = value
            
            # 5. שמירת הנתונים לקובץ חדש אם מצאנו נתונים רלוונטיים
            if patient_data:
                output_filepath = os.path.join(output_dir, f"{patient_id}.json")
                with open(output_filepath, 'w') as out_file:
                    # שימוש ב-indent=4 כדי שהקובץ יהיה קריא ונוח לעין
                    json.dump(patient_data, out_file, indent=4)
                print(f"Saved clean data for patient {patient_id} -> {output_filepath}")

# הרצת הקוד: 
# יש לוודא שהסקריפט רץ באותה תיקייה שבה נמצאים קבצי ה-.hea של המטופלים
if __name__ == "__main__":
    create_clean_info_folder()