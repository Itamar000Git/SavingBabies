import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random  # הוספנו ספרייה לקיבוע אקראיות
import matplotlib.pyplot as plt


# ==========================================
# 0. קיבוע אקראיות (Seed Fix) - חדש!
# ==========================================
# זה מבטיח שאם תריץ את הקוד 10 פעמים, תקבל תוצאות זהות ויציבות, ונוכל לבודד באגים.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ==========================================
# 1. Settings and paths
# ==========================================
csv_dir = "csv_output"
ph_file_path = "dataset/ph_levels.csv"
SEQ_LEN = 1200
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Load real signals from CSV files
# ==========================================
print("Loading real data from CSV files...")

ph_df = pd.read_csv(ph_file_path)
ph_df['record_id'] = ph_df['record_id'].astype(str)
ph_dict = dict(zip(ph_df['record_id'], ph_df['pH']))

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
X_list = []
y_list_numeric = []

for file in csv_files:
    record_id = os.path.basename(file).split('.')[0]

    if record_id not in ph_dict:
        continue

    try:
        df = pd.read_csv(file)
        df['FHR'] = df['FHR'].replace(0, np.nan).interpolate(method='linear').fillna(140)
        df['UC'] = df['UC'].replace(0, np.nan).interpolate(method='linear').fillna(0)

        seq = np.column_stack((df['FHR'].values, df['UC'].values))

        if len(seq) >= SEQ_LEN:
            seq = seq[-SEQ_LEN:]
        else:
            pad_len = SEQ_LEN - len(seq)
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')

        X_list.append(seq)
        y_list_numeric.append(ph_dict[record_id])

    except Exception as e:
        print(f"Failed to process {file}: {e}")

X_data = np.array(X_list)
y_data_numeric = np.array(y_list_numeric)

print(f"Successfully loaded {len(X_data)} records.")


# ==========================================
# 3. Convert pH into clinical classes + split data
# ==========================================
def classify_ph_acog(ph_array):
    conditions = [
        ph_array < 7.10,
        (ph_array >= 7.10) & (ph_array < 7.20),
        ph_array >= 7.20
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices)


y_data_classes = classify_ph_acog(y_data_numeric)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_data, y_data_classes,
    test_size=0.15,
    random_state=42,
    stratify=y_data_classes
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15,
    random_state=42,
    stratify=y_temp
)

# ==========================================
# 3.5 תיקון זליגת המידע (Data Leakage Fix) - חדש!
# ==========================================
# מחשבים את הממוצע וסטיית התקן *רק* על סט האימון.
# בעבר, המודל "הציץ" לנתוני העתיד (Val/Test) וחישב על פיהם. עכשיו זה נקי.
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0) + 1e-8

# מנרמלים את כל הקבוצות לפי המדדים של האימון בלבד
X_train_norm = (X_train - train_mean) / train_std
X_val_norm = (X_val - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std


class CTGClassificationDataset(Dataset):
    def __init__(self, X_normalized, y):
        # ה-Dataset כבר מקבל את הנתונים המנורמלים מבחוץ, אז פשוט ממירים לטנזור
        self.X = torch.tensor(X_normalized, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# העברת הנתונים המנורמלים החדשים ל-Loaders
train_loader = DataLoader(CTGClassificationDataset(X_train_norm, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CTGClassificationDataset(X_val_norm, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CTGClassificationDataset(X_test_norm, y_test), batch_size=1, shuffle=False)


# ==========================================
# 4. Model architecture (CNN + Attention) + מנגנוני ייצוב
# ==========================================
class HybridCTGClassifier(nn.Module):
    def __init__(self, input_dim=2, embed_dim=16, num_heads=4, num_classes=3):
        super(HybridCTGClassifier, self).__init__()

        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=15, padding=7)

        # תוספת חדשה: נרמול אצוות (Batch Normalization)
        # מייצב את הערכים שיוצאים מה-CNN ומונע "פיצוץ" של מספרים במהלך האימון
        self.bn = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.fc = nn.Sequential(
            # תוספת חדשה: Dropout (30%)
            # "מכבה" אקראית 30% מהנוירונים באימון. מכריח את הרשת לא לשנן נתונים בעל-פה.
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            # תוספת חדשה: עוד Dropout קטן (20%) לפני ההחלטה הסופית
            nn.Dropout(p=0.2),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        # ה-CNN מעביר את הפלט ל-BatchNorm ורק אז לאקטיבציה (ReLU)
        cnn_out = self.relu(self.bn(self.cnn(x)))
        cnn_out = cnn_out.transpose(1, 2)

        attn_output, attn_weights = self.attention(cnn_out, cnn_out, cnn_out, need_weights=True)
        pooled = attn_output.mean(dim=1)
        out = self.fc(pooled)
        return out, attn_weights


model = HybridCTGClassifier().to(DEVICE)

# משקלים לסיווג - אני משאיר את מה שהיה לך: תעדוף ענק למחלקה המסוכנת
class_weights = torch.tensor([7.0, 5.0, 1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# הלרנינג רייט נשאר 0.001 קבוע לבקשתך
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. Training loop
# ==========================================
print("\n--- Starting Classification Training ---")
for epoch in range(EPOCHS):
    model.train()  # במצב Train, ה-Dropout מופעל פעיל ומכבה נוירונים!
    running_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        predictions, _ = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    model.eval()  # במצב Eval, ה-Dropout נכבה והרשת משתמשת בכל הנוירונים שלה לבדיקה
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
# 6. Evaluation on test set
# ==========================================
print("\n--- Evaluating on Test Set ---")
model.eval()

all_true_classes = []
all_pred_classes = []

with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch = X_test_batch.to(DEVICE)
        pred_logits, _ = model(X_test_batch)
        predicted_class = torch.argmax(pred_logits, dim=1)

        all_true_classes.append(y_test_batch.item())
        all_pred_classes.append(predicted_class.item())

print("\nClinical Classification Report:")
target_names = ["Dangerous", "Intermediate", "Healthy"]

present_classes = np.unique(np.concatenate((all_true_classes, all_pred_classes)))
present_target_names = [target_names[i] for i in present_classes]

print(classification_report(
    all_true_classes, all_pred_classes,
    target_names=present_target_names,
    labels=present_classes,
    zero_division=0
))