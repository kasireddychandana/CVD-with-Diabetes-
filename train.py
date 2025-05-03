import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score



# --- Load Dataset ---
df = pd.read_csv('data/framingham.csv')
df = df.dropna()
print("✅ Dataset loaded")

# --- Feature Selection ---
features = ['glucose', 'diabetes', 'age', 'BMI', 'sysBP', 'diaBP', 'totChol', 'cigsPerDay']
X = df[features]
y = df['TenYearCHD']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Done train_test_split")

# --- Apply SMOTE for Imbalance ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("✅ Applied SMOTE for class imbalance")

# --- Grid Search for XGBoost ---
param_grid_xgb = {
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'n_estimators': [100],
    'subsample': [0.8],
    'colsample_bytree': [0.7]
}
grid_xgb = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                        param_grid_xgb, scoring='accuracy', cv=3, verbose=1)
grid_xgb.fit(X_train_res, y_train_res)
best_xgb = grid_xgb.best_estimator_
print("✅ XGBoost model trained")

# --- Grid Search for CatBoost ---
param_grid_cat = {
    'learning_rate': [0.1],
    'depth': [6, 8],
    'iterations': [100],
    'l2_leaf_reg': [3]
}
grid_cat = GridSearchCV(CatBoostClassifier(verbose=0),
                        param_grid_cat, scoring='accuracy', cv=3, verbose=1)
grid_cat.fit(X_train_res, y_train_res)
best_cat = grid_cat.best_estimator_
print("✅ CatBoost model trained")

# --- Calibration ---
cal_xgb = CalibratedClassifierCV(best_xgb, method='sigmoid', cv='prefit')
cal_cat = CalibratedClassifierCV(best_cat, method='sigmoid', cv='prefit')
cal_xgb.fit(X_train_res, y_train_res)
cal_cat.fit(X_train_res, y_train_res)

# --- Predictions ---
xgb_probs = cal_xgb.predict_proba(X_test)[:, 1]
cat_probs = cal_cat.predict_proba(X_test)[:, 1]

# --- Meta-model Input ---
X_meta = np.column_stack((xgb_probs, cat_probs))
meta_model = LogisticRegression()
meta_model.fit(X_meta, y_test)

# --- Final Meta-model Prediction ---
final_probs = meta_model.predict_proba(X_meta)[:, 1]
final_preds = (final_probs >= 0.5).astype(int)

print("\n📊 Final Stacked Model Performance (Threshold = 0.5):")
print(classification_report(y_test, final_preds))

# --- Threshold Tuning ---
threshold = 0.3
final_preds_thresh = (final_probs >= threshold).astype(int)

print(f"\n📊 Final Model Performance with Threshold Adjustment (Threshold = {threshold}):")
print(classification_report(y_test, final_preds_thresh))

# --- Optional: Threshold Loop ---
print("\n🔁 Precision/Recall for Thresholds:")
for t in np.arange(0.1, 0.9, 0.05):
    preds_t = (final_probs >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds_t, average='binary')
    print(f"Threshold: {t:.2f} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")

# --- Save Models ---
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open('cat_model.pkl', 'wb') as f:
    pickle.dump(best_cat, f)

with open('meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)

print("\n✅ All models saved successfully.")
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    preds_t = (final_probs >= t).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_test, preds_t, average='binary', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n🏆 Best Threshold based on F1 score: {best_threshold:.2f} (F1 = {best_f1:.2f})")

# --- Save Best Threshold for Future Use ---



# Save to file
with open('best_threshold.txt', 'w') as f:
    rounded_threshold = round(best_threshold, 2)
    f.write(str(rounded_threshold))


print("\n📁 Saved best threshold to 'best_threshold.txt'")

conf_matrix = confusion_matrix(y_test, final_preds_thresh)
print(f"\n📊 Confusion Matrix:\n{conf_matrix}")
tn, fp, fn, tp = confusion_matrix(y_test, final_preds_thresh).ravel()
specificity = tn / (tn + fp)
print(f"✅ Specificity: {specificity:.2f}")

logloss = log_loss(y_test, final_probs)
print(f"✅ Log-Loss (Cross-Entropy): {logloss:.2f}")


average_precision = average_precision_score(y_test, final_probs)
print(f"✅ Average Precision-Recall AUC: {average_precision:.2f}")

