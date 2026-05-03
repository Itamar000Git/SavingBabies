import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pyplot as plt

# ==========================================
# 1. הגדרות ונתיבים (מותאם לצילום המסך שלך)
# ==========================================
# שים לב: אם אתה מריץ את הקוד מתוך התיקייה הראשית (SavingBabies), הנתיבים האלו יעבדו.
# אם הקוד לא מוצא את הקבצים, שנה את csv_dir ל: "../csv_output" ואת ph_file ל: "ph_levels.csv"
csv_dir = "csv_output"
ph_file_path = "dataset/ph_levels.csv"

SEQ_LEN = 1200  # חלון הזמן (5 דקות ב-4Hz)
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. חילוץ אמיתי של נתוני ה-CSV
# ==========================================
print("Loading real data from CSV files...")

ph_df = pd.read_csv(ph_file_path)
ph_df['record_id'] = ph_df['record_id'].astype(str)
ph_dict = dict(zip(ph_df['record_id'], ph_df['pH']))

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
X_list = []
y_list = []

for file in csv_files:
    record_id = os.path.basename(file).split('.')[0]

    # אם אין לנו pH עבור ההקלטה הזו, נדלג עליה
    if record_id not in ph_dict:
        continue

    try:
        df = pd.read_csv(file)

        # ניקוי רעשים ואינטרפולציה
        df['FHR'] = df['FHR'].replace(0, np.nan).interpolate(method='linear').fillna(140)
        df['UC'] = df['UC'].replace(0, np.nan).interpolate(method='linear').fillna(0)

        seq = np.column_stack((df['FHR'].values, df['UC'].values))

        # חיתוך או ריפוד
        if len(seq) >= SEQ_LEN:
            seq = seq[-SEQ_LEN:]
        else:
            pad_len = SEQ_LEN - len(seq)
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')

        X_list.append(seq)
        y_list.append(ph_dict[record_id])

    except Exception as e:
        print(f"Failed to process {file}: {e}")

X_data = np.array(X_list)
y_data = np.array(y_list)

print(f"Successfully loaded {len(X_data)} records.")


# ==========================================
# 3. פונקציות סיווג ושגיאה מותאמת אישית (Weighted Loss)
# ==========================================
def classify_ph_numeric(ph_array):
    conditions = [ph_array < 7.10, (ph_array >= 7.10) & (ph_array < 7.20), ph_array >= 7.20]
    choices = [0, 1, 2]  # 0=Dangerous, 1=Intermediate, 2=Healthy
    return np.select(conditions, choices)


def get_class_name(class_idx):
    mapping = {0: "Dangerous (Critical)", 1: "Intermediate (Monitor)", 2: "Healthy (Normal)"}
    return mapping.get(class_idx, "Unknown")


# הפתרון לחוסר איזון! פונקציה שנותנת משקל לטעות לפי חומרת המצב
class WeightedMSELoss(nn.Module):
    def __init__(self, critical_w=10.0, inter_w=2.0, healthy_w=1.0):
        super().__init__()
        self.critical_w = critical_w
        self.inter_w = inter_w
        self.healthy_w = healthy_w

    def forward(self, pred, target):
        se = (pred - target) ** 2  # השגיאה הרגילה

        # יצירת וקטור משקלים לפי ה-pH האמיתי
        weights = torch.ones_like(target)
        weights[target < 7.10] = self.critical_w
        weights[(target >= 7.10) & (target < 7.20)] = self.inter_w
        weights[target >= 7.20] = self.healthy_w

        # הכפלת השגיאה במשקל
        weighted_se = se * weights
        return weighted_se.mean()


# ==========================================
# 4. הכנת הנתונים (Dataset & DataLoader)
# ==========================================
X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)


class CTGDataset(Dataset):
    def __init__(self, X, y):
        # Standardization
        self.X = torch.tensor((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(CTGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CTGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CTGDataset(X_test, y_test), batch_size=1, shuffle=False)


# ==========================================
# 5. ארכיטקטורת המודל
# ==========================================
class AttentionCTGTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=32, num_heads=4):
        super(AttentionCTGTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        embedded = self.input_projection(x)
        attn_output, attn_weights = self.attention(embedded, embedded, embedded, need_weights=True)
        pooled = attn_output.mean(dim=1)
        out = self.fc(pooled)
        return out.squeeze(), attn_weights


model = AttentionCTGTransformer().to(DEVICE)
criterion = WeightedMSELoss(critical_w=15.0, inter_w=3.0, healthy_w=1.0)  # משקל כבד לטעות על חמצת!
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # קצב למידה טיפה יותר נמוך ליציבות

# ==========================================
# 6. לולאת אימון עם Validation
# ==========================================
print("\n--- Starting Training on Real Data ---")
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        predictions, _ = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
            val_preds, _ = model(X_val_batch)
            v_loss = criterion(val_preds, y_val_batch)
            running_val_loss += v_loss.item()

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {running_train_loss / len(train_loader):.4f} | Val Loss: {running_val_loss / len(val_loader):.4f}")

# ==========================================
# 7. בדיקת ביצועים וסיווג (Testing)
# ==========================================
print("\n--- Evaluating on Test Set ---")
model.eval()
all_true_ph = []
all_pred_ph = []

with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch = X_test_batch.to(DEVICE)
        pred, _ = model(X_test_batch)
        all_true_ph.append(y_test_batch.item())
        all_pred_ph.append(pred.item())

all_true_ph = np.array(all_true_ph)
all_pred_ph = np.array(all_pred_ph)

true_classes = classify_ph_numeric(all_true_ph)
pred_classes = classify_ph_numeric(all_pred_ph)

# דוח מפורט - כאן תראה את ההשפעה של המשקלים!
print("\nClinical Classification Report:")
target_names = ["Dangerous", "Intermediate", "Healthy"]

# בודקים אילו מחלקות באמת קיימות ב-Test set כדי לא לקבל שגיאה בהדפסה
present_classes = np.unique(np.concatenate((true_classes, pred_classes)))
present_target_names = [target_names[i] for i in present_classes]

print(classification_report(true_classes, pred_classes,
                            target_names=present_target_names,
                            labels=present_classes,
                            zero_division=0))