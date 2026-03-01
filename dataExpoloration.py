import os
import glob
import numpy as np
import pandas as pd
import wfdb

# הגדרת נתיבי התיקיות
data_dir = "dataset"  # התיקייה שבה נמצאים קבצי המקור (.dat, .hea)
output_dir = "csv_output"  # התיקייה שבה יישמרו קבצי ה-CSV החדשים

# יצירת תיקיית הפלט במידה והיא לא קיימת
os.makedirs(output_dir, exist_ok=True)

# מציאת כל קבצי ה-.hea בתיקיית הנתונים
# זה יעזור לנו למצוא את כל מספרי הרשומות הקיימות
hea_files = glob.glob(os.path.join(data_dir, "*.hea"))

if not hea_files:
    print(f"No .hea files found in the '{data_dir}' directory. Please check your folder structure.")

# iterate across all the files
for hea_file in hea_files:
    # extract the file name
    record_name = os.path.splitext(os.path.basename(hea_file))[0]

    # full path
    record_path = os.path.join(data_dir, record_name)

    try:
        # get the current record
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        fs = record.fs

        # creating time line
        t = np.arange(signals.shape[0]) / fs

        # creating the data frame
        df = pd.DataFrame({
            "t_sec": t,
            "FHR": signals[:, 0],
            "UC": signals[:, 1],
        })

        # save to the csv
        csv_filename = os.path.join(output_dir, f"{record_name}.csv")
        df.to_csv(csv_filename, index=False)

        print(f"Successfully processed: {record_name} -> {csv_filename}")

    except Exception as e:
        print(f"Error processing record {record_name}: {e}")

print("Done processing all files!")