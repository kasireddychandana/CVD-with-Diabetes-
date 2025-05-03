import pickle
import numpy as np

# --- Load Models ---
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('cat_model.pkl', 'rb') as f:
    cat_model = pickle.load(f)

with open('meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)

with open('best_threshold.txt', 'r') as f:
    best_threshold = float(f.read().strip())

print("✅ Models and threshold loaded")

# --- Example New Patient Input (Must match feature order) ---
# Format: [glucose, diabetes, age, BMI, sysBP, diaBP, totChol, cigsPerDay]
new_patient = np.array([[85, 1, 52, 26.7, 140, 90, 210, 5]])

# --- Get probabilities from base models ---
xgb_prob = xgb_model.predict_proba(new_patient)[:, 1]
cat_prob = cat_model.predict_proba(new_patient)[:, 1]

# --- Stack and Predict with Meta-model ---
meta_input = np.column_stack((xgb_prob, cat_prob))
final_prob = meta_model.predict_proba(meta_input)[:, 1]

# --- Apply Threshold ---
prediction = int(final_prob[0] >= best_threshold)

# --- Output Result ---
print(f"\n🔎 Final Probability of CHD: {final_prob[0]:.2f}")
print(f"📣 Prediction: {'High Risk (CHD)' if prediction == 1 else 'Low Risk (No CHD)'}")
