"""
训练并保存模型 + 准备数据库
"""
import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

CLEAN_CSV = "data/steam_games_clean.csv"
DB_PATH = "data/steam.db"
MODEL_PATH = "data/model_rf.pkl"
ENCODER_PATH = "data/label_encoders.pkl"
FEATURES_PATH = "data/feature_columns.pkl"

print("[1/4] Loading clean data...")
df = pd.read_csv(CLEAN_CSV, encoding='utf-8-sig')
df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
print(f"  {len(df)} records")

# ====== Feature columns ======
tag_cols = [c for c in df.columns if c.startswith('Tag_')]
feature_cols = tag_cols + [
    'OriginalPrice', 'DiscountRate', 'IsFree', 'IsOnSale',
    'ReleaseYear', 'ReleaseMonth',
    'NumFeatures', 'NumTags', 'NumLanguages',
    'HasSinglePlayer', 'HasMultiplayer', 'HasControllerSupport',
    'HasAchievements', 'HasTradingCards', 'HasInAppPurchases', 'HasWorkshop',
    'SupportsChinese',
]
feature_cols = [c for c in feature_cols if c in df.columns]

# ====== Encode Developer/Publisher ======
print("[2/4] Encoding categorical features...")
from sklearn.preprocessing import LabelEncoder

le_dev = LabelEncoder()
le_pub = LabelEncoder()

# Fill NaN
df['Developer'] = df['Developer'].fillna('Unknown')
df['Publisher'] = df['Publisher'].fillna('Unknown')

# Only encode if enough samples, else mark as 'Other'
dev_counts = df['Developer'].value_counts()
df['DevEncoded'] = le_dev.fit_transform(
    df['Developer'].apply(lambda x: x if dev_counts.get(x, 0) >= 5 else 'Other')
)

pub_counts = df['Publisher'].value_counts()
df['PubEncoded'] = le_pub.fit_transform(
    df['Publisher'].apply(lambda x: x if pub_counts.get(x, 0) >= 5 else 'Other')
)

# ====== Train regression model ======
print("[3/4] Training Random Forest...")
reg_df = df.dropna(subset=['AllReviewPct']).copy()
all_features = feature_cols + ['DevEncoded', 'PubEncoded']
X = reg_df[all_features].fillna(0)
y = reg_df['AllReviewPct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"  R2 score: {score:.4f}")

# Feature importance
importances = dict(zip(all_features, model.feature_importances_))

# ====== Save everything ======
print("[4/4] Saving model + database...")
os.makedirs("data", exist_ok=True)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(FEATURES_PATH, 'wb') as f:
    pickle.dump({'feature_cols': all_features, 'importances': importances}, f)

# ====== Build SQLite database ======
conn = sqlite3.connect(DB_PATH)
df.to_sql('games', conn, if_exists='replace', index=True, index_label='id')
# Rebuild with proper rowid as id
conn.execute("DROP TABLE IF EXISTS games")
df.reset_index(drop=True).to_sql('games', conn, if_exists='replace', index_label='id')

# Create indices
conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON games(OriginalPrice)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON games(AllReviewPct)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON games(ReleaseYear)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_dev ON games(Developer)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON games(NumTags)")

conn.commit()
conn.close()

print(f"  Model saved to: {MODEL_PATH}")
print(f"  Database saved to: {DB_PATH}")
print(f"  Total games in DB: {len(df)}")
print("\nDone! Ready to run app.py")
