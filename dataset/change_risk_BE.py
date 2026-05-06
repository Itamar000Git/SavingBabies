import pandas as pd

# טעינת הקובץ
df = pd.read_csv('ph_levels_with_be_risk_filled.csv')

# עדכון השדה riskBE: יהיה 1 אם BE גדול מ-10, אחרת 0
df['riskBE'] = df['BE'].apply(lambda x: 1 if x > 12 else 0)

# שמירת הקובץ המעודכן
df.to_csv('updated_ph_levels_risk.csv', index=False)

print("הקובץ עודכן ונשמר בהצלחה!")