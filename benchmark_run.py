import numpy as np
from robust_bayes import CertifiedRobustBayes

# 1. Generate Synthetic Data (Simulating AI vs Human text detection)
np.random.seed(42)
n_features = 50
n_samples = 100

# Class 0: Human, Class 1: AI (AI has more occurrences of first 5 features)
X = np.random.poisson(lam=1.0, size=(n_samples, n_features))
X[50:, :5] += 3 
y = np.array([0]*50 + [1]*50)

# 2. Initialize and Train
model = CertifiedRobustBayes(alpha=1.0)
model.fit(X, y)

# 3. Select a test sample (an AI-generated text)
test_sample = X[60:61]
adversarial_budget = 2  # Attacker can flip 2 words

pred, certified, margin = model.certify_robustness(test_sample, budget=adversarial_budget)

# 4. Print results
print("--- ROBUSTNESS VERIFICATION REPORT ---")
print(f"Prediction: {'AI-Generated' if pred == 1 else 'Human-Authored'}")
print(f"Adversarial Budget: {adversarial_budget} words")
print(f"Is Prediction Certified Robust? {'YES' if certified else 'NO'}")
print(f"Safe Margin (Stability): {margin:.4f}")
