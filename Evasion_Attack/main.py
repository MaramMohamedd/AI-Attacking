from preprocessing import load_and_clean_data
from model import train_model
from attack_evasion import evasion_attack, perturb_text
from defense import defended_vectorize, ensemble_voting

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# ── Load & Train ──────────────────────────────────────────────────────────────
df = load_and_clean_data("data/spam.csv")
model, vectorizer, X_train, X_test, y_train, y_test = train_model(df)

# ── BEFORE ATTACK ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── AFTER ATTACK ──────────────────────────────────────────────────────────────
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
test_texts_orig = df_test['clean_text'].tolist()
test_texts_pert = [perturb_text(t) for t in test_texts_orig]
y_test_true     = df_test['label'].values

X_test_pert     = vectorizer.transform(test_texts_pert)
y_pred_attacked = model.predict(X_test_pert)

# ── AFTER DEFENSE 1 ───────────────────────────────────────────────────────────
X_test_sanitized = defended_vectorize(test_texts_pert, vectorizer)
y_pred_def1      = model.predict(X_test_sanitized)

# ── AFTER DEFENSE 2 ───────────────────────────────────────────────────────────
model1, model2, model3 = ensemble_voting(df_train, vectorizer)

X_sanitized_pert = defended_vectorize(test_texts_pert, vectorizer)
pred1 = model1.predict(X_sanitized_pert)
pred2 = model2.predict(X_sanitized_pert)
pred3 = model3.predict(X_sanitized_pert)

votes = np.stack([pred1, pred2, pred3], axis=1)
y_pred_def2 = (votes.sum(axis=1) >= 2).astype(int)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("=" * 60)
print("                    RESULTS SUMMARY")
print("=" * 60)
print(f"{'Step':<30} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 60)

rows = [
    ("Before Attack",            y_test,      y_pred),
    ("After Attack",             y_test_true, y_pred_attacked),
    ("Defense 1: Sanitization",  y_test_true, y_pred_def1),
    ("Defense 2: Ensemble Vote", y_test_true, y_pred_def2),
]
for name, yt, yp in rows:
    print(f"{name:<30} "
          f"{accuracy_score(yt,yp):>6.4f} "
          f"{precision_score(yt,yp):>7.4f} "
          f"{recall_score(yt,yp):>7.4f} "
          f"{f1_score(yt,yp):>7.4f}")
print("=" * 60)