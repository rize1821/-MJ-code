
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ── 1. 시드 고정 ────────────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ── 2. 경로 설정 & 데이터 로드 ─────────────────────────────────────
BASE = os.path.dirname(__file__)
train_X = pd.read_csv(os.path.join(BASE, 'train_X.csv'))
train_y = pd.read_csv(os.path.join(BASE, 'train_y.csv'))
val_X   = pd.read_csv(os.path.join(BASE, 'validation_X.csv'))
val_y   = pd.read_csv(os.path.join(BASE, 'validation_y.csv'))
test_X  = pd.read_csv(os.path.join(BASE, 'test_X.csv'))

# ── 3. 추가 피처 엔지니어링 ─────────────────────────────────────────
for df in (train_X, val_X, test_X):
    df.sort_values(['vehicle_id','time_step'], inplace=True)
    df['delta_y']     = df.groupby('vehicle_id')['position_y'].diff().fillna(0)
    df['delta_speed'] = df.groupby('vehicle_id')['speed'].diff().fillna(0)
    df['delta_acc']   = df.groupby('vehicle_id')['acceleration'].diff().fillna(0)
features = ['position_y','speed','acceleration','delta_y','delta_speed','delta_acc']

# ── 4. 시퀀스 빌드 + 패딩 ───────────────────────────────────────────
def build_seqs(df, ids=None):
    if ids is None:
        ids = pd.unique(df['vehicle_id']).tolist()
    seqs = []
    for vid in ids:
        sub = df[df['vehicle_id']==vid].sort_values('time_step')
        seqs.append(sub[features].values)
    T = max(s.shape[0] for s in seqs)
    padded = []
    for s in seqs:
        if s.shape[0] < T:
            pad = np.zeros((T - s.shape[0], len(features)))
            padded.append(np.vstack([s, pad]))
        else:
            padded.append(s)
    return np.stack(padded), [str(v) for v in ids]

X_tr, tr_ids = build_seqs(train_X)
y_tr = train_y.set_index('vehicle_id').loc[tr_ids,'change_section'].values.astype(int)
X_va, va_ids = build_seqs(val_X)
y_va = val_y.set_index('vehicle_id').loc[va_ids,'change_section'].values.astype(int)
X_te, te_ids = build_seqs(test_X)

# ── 5. 정규화 ─────────────────────────────────────────────────────────
feat_dim = X_tr.shape[2]
scaler   = StandardScaler().fit(X_tr.reshape(-1, feat_dim))
X_tr = scaler.transform(X_tr.reshape(-1, feat_dim)).reshape(X_tr.shape)
X_va = scaler.transform(X_va.reshape(-1, feat_dim)).reshape(X_va.shape)
X_te = scaler.transform(X_te.reshape(-1, feat_dim)).reshape(X_te.shape)

# ── 6. 클래스 불균형 처리 ─────────────────────────────────────────────
class_counts   = np.bincount(y_tr)
class_weights  = 1.0 / class_counts
sample_weights = class_weights[y_tr]
train_sampler  = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ── 7. Dataset 정의 ───────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, X, y=None, ids=None):
        self.X, self.y, self.ids = torch.tensor(X, dtype=torch.float32), y, ids
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        if self.y is not None:
            return self.X[i], self.y[i]
        return self.X[i], self.ids[i]

train_dl = DataLoader(SeqDS(X_tr, y_tr), batch_size=64, sampler=train_sampler)
val_dl   = DataLoader(SeqDS(X_va, y_va), batch_size=64, shuffle=False)
test_dl  = DataLoader(SeqDS(X_te, ids=te_ids), batch_size=64, shuffle=False)

# ── 8. 모델 정의 ─────────────────────────────────────────────────────
class BiLSTMAttn(nn.Module):
    def __init__(self, feat_dim, hid_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(feat_dim, hid_dim, num_layers=n_layers,
                               bidirectional=True, batch_first=True, dropout=dropout)
        self.attn_fc = nn.Linear(2*hid_dim, 1)
        self.fc      = nn.Linear(2*hid_dim, 4)
    def forward(self, x):
        h, _ = self.lstm(x)
        w    = torch.softmax(torch.tanh(self.attn_fc(h)), dim=1)
        z    = (h * w).sum(dim=1)
        return self.fc(z)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = BiLSTMAttn(feat_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                factor=0.5, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

# ── 9. 학습 & 과적합 모니터링 ─────────────────────────────────────────
train_losses, val_losses = [], []
best_f1, no_imp = 0.0, 0

for epoch in range(1, 31):
    # Train
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_dl)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    total_val_loss, preds, labs = 0.0, [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_val_loss += criterion(out, yb).item()
            preds.extend(out.argmax(1).cpu().numpy())
            labs.extend(yb.cpu().numpy())
    val_loss = total_val_loss / len(val_dl)
    val_losses.append(val_loss)

    # Metrics
    f1   = f1_score(labs, preds, average='weighted')
    acc  = accuracy_score(labs, preds)
    prec = precision_score(labs, preds, average='weighted', zero_division=0)
    rec  = recall_score(labs, preds, average='weighted', zero_division=0)

    scheduler.step(f1)
    print(f"Epoch {epoch:2d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} "
          f"| Acc {acc:.4f} Prec {prec:.4f} Rec {rec:.4f} F1 {f1:.4f}")

    if f1 > best_f1:
        best_f1, no_imp = f1, 0
        torch.save(model.state_dict(), os.path.join(BASE, 'best_model.pt'))
    else:
        no_imp += 1
        if no_imp >= 5:
            print(f"EarlyStopping at epoch {epoch}, best F1 {best_f1:.4f}")
            break

# ── 10. 과적합 모니터링: 손실 곡선 ───────────────────────────────────
plt.figure(figsize=(6,4))
plt.plot(range(1,len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1,len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')
plt.show()

# ── 11. 최종 검증 지표 ───────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(BASE, 'best_model.pt'), map_location=device))
model.eval()
vpreds, vlabs = [], []
with torch.no_grad():
    for xb, yb in val_dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        vpreds.extend(out.argmax(1).cpu().numpy())
        vlabs.extend(yb.cpu().numpy())

print("\n**최종 검증 지표**")
print(f"Accuracy : {accuracy_score(vlabs, vpreds):.4f}")
print(f"Precision: {precision_score(vlabs, vpreds, average='weighted'):.4f}")
print(f"Recall   : {recall_score(vlabs, vpreds, average='weighted'):.4f}")
print(f"F1-Score : {f1_score(vlabs, vpreds, average='weighted'):.4f}")

# ── 12. 테스트 예측 & submission.csv 생성 ─────────────────────────────
rows = []
with torch.no_grad():
    for xb, ids in test_dl:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu().numpy()
        rows.extend([(vid, str(int(p))) for vid,p in zip(ids, preds)])

pd.DataFrame(rows, columns=['vehicle_id','change_section'])\
  .to_csv(os.path.join(BASE, 'submission.csv'), index=False)

print("✔ submission.csv 생성 완료:", os.path.join(BASE, 'submission.csv'))
