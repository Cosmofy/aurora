"""
Dual Model System: GB + XGBoost with agreement-based confidence

Logic:
- Both agree high → high confidence
- Both agree low → high confidence (no aurora)
- They disagree → use average, flag as uncertain
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DUAL MODEL SYSTEM")
print("=" * 70)

# Load and prep data
df = pd.read_csv("data/training.csv")
df = df[(df['aurora'] == 1) | (df['latitude'].abs() >= 40)]
df = df.drop(columns=["datetime", "year", "windgust", "uvindex", "solarradiation", "visibility"])

# Feature engineering
df['is_dark'] = (df['sun_altitude'] < -6).astype(int)
df['moon_interference'] = ((df['moon_altitude'] > 0) & (df['moon_illumination'] > 0.5)).astype(int)
df['storm'] = (df['kp_index'] >= 5).astype(int)
df['strong_storm'] = (df['dst'] < -50).astype(int)
df['lat_kp'] = df['magnetic_latitude'] * df['kp_index']
df['dark_storm'] = df['is_dark'] * df['storm']
df['good_conditions'] = ((df['cloudcover'] < 50) & (df['sun_altitude'] < -6)).astype(int)
df['sw_pressure'] = df['solar_wind_speed'] * df['solar_wind_density']

X = df.drop(columns=["aurora"])
X = X.fillna(X.median())
y = df["aurora"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

# Train both models
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(
    random_state=42,
    max_depth=3,
    n_estimators=200,
    learning_rate=0.05,
    min_samples_split=10,
    min_samples_leaf=5
)
gb.fit(X_train, y_train)

print("Training XGBoost...")
xgb = XGBClassifier(
    random_state=42,
    max_depth=4,
    n_estimators=200,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# Test dual system
print("\n" + "=" * 70)
print("DUAL MODEL RESULTS")
print("=" * 70)

gb_proba = gb.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]

# Agreement analysis
diff = np.abs(gb_proba - xgb_proba)
avg_proba = (gb_proba + xgb_proba) / 2

print(f"\nModel agreement stats:")
print(f"  Mean probability difference: {diff.mean():.3f}")
print(f"  Max difference: {diff.max():.3f}")
print(f"  Samples with >20% disagreement: {(diff > 0.2).sum()} ({(diff > 0.2).mean()*100:.1f}%)")
print(f"  Samples with >10% disagreement: {(diff > 0.1).sum()} ({(diff > 0.1).mean()*100:.1f}%)")

# Confidence levels
def get_confidence(gb_p, xgb_p):
    avg = (gb_p + xgb_p) / 2
    diff = abs(gb_p - xgb_p)

    if diff < 0.1:
        agreement = "strong"
    elif diff < 0.2:
        agreement = "moderate"
    else:
        agreement = "weak"

    return avg, agreement

# Test on a few samples
print(f"\n{'='*70}")
print("SAMPLE PREDICTIONS")
print("=" * 70)
print(f"{'GB':>8} {'XGB':>8} {'Avg':>8} {'Agree':>10} {'Actual':>8}")
print("-" * 50)

for i in range(min(20, len(X_test))):
    avg, agreement = get_confidence(gb_proba[i], xgb_proba[i])
    actual = "AURORA" if y_test.iloc[i] == 1 else "no"
    print(f"{gb_proba[i]:>7.1%} {xgb_proba[i]:>7.1%} {avg:>7.1%} {agreement:>10} {actual:>8}")

# Accuracy at different confidence levels
print(f"\n{'='*70}")
print("ACCURACY BY AGREEMENT LEVEL")
print("=" * 70)

for threshold in [0.1, 0.15, 0.2]:
    strong_agree = diff < threshold
    if strong_agree.sum() > 0:
        # Use average probability with 0.5 cutoff
        preds = (avg_proba >= 0.5).astype(int)
        acc = (preds[strong_agree] == y_test.values[strong_agree]).mean()
        print(f"Agreement <{threshold:.0%}: {strong_agree.sum()} samples ({strong_agree.mean()*100:.1f}%), accuracy: {acc:.1%}")

# Overall accuracy
preds = (avg_proba >= 0.5).astype(int)
overall_acc = (preds == y_test.values).mean()
print(f"\nOverall (averaged): {overall_acc:.1%}")
print(f"GB alone: {gb.score(X_test, y_test):.1%}")
print(f"XGB alone: {xgb.score(X_test, y_test):.1%}")

# Save both models
print(f"\n{'='*70}")
print("SAVING MODELS")
print("=" * 70)

import os
os.makedirs('models', exist_ok=True)
joblib.dump(gb, 'models/gb.joblib')
joblib.dump(xgb, 'models/xgb.joblib')

print(f"Saved: models/gb.joblib ({gb.__class__.__name__})")
print(f"Saved: models/xgb.joblib ({xgb.__class__.__name__})")

print(f"\n{'='*70}")
print("INFERENCE EXAMPLE")
print("=" * 70)
print("""
# Load models
gb = joblib.load('model_gb.joblib')
xgb = joblib.load('model_xgb.joblib')

# Get probabilities
gb_prob = gb.predict_proba(X)[0][1]
xgb_prob = xgb.predict_proba(X)[0][1]

# Combine
avg_prob = (gb_prob + xgb_prob) / 2
diff = abs(gb_prob - xgb_prob)

if diff < 0.1:
    confidence = "high"
elif diff < 0.2:
    confidence = "moderate"
else:
    confidence = "low"

# Return to user
return {
    "probability": avg_prob,
    "confidence": confidence,
    "gb_says": gb_prob,
    "xgb_says": xgb_prob
}
""")
