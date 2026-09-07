import numpy as np
import matplotlib.pyplot as plt
from robust_bayes import CertifiedRobustBayes

# 1. Генериране на изкуствени данни (Симулация на AI срещу човешки текст)
np.random.seed(42)
n_features = 50
n_samples = 100

# Клас 0: Човек, Клас 1: AI (AI съдържа по-често първите 5 специфични думи/маркери)
X = np.random.poisson(lam=1.0, size=(n_samples, n_features))
X[50:, :5] += 3 
y = np.array([0]*50 + [1]*50)

# 2. Инициализиране и обучение на модела
model = CertifiedRobustBayes(alpha=1.0)
model.fit(X, y)

# 3. Избор на тестова извадка (AI-генериран текст)
test_sample = X[60:61]

# Тестване на конкретен единичен сценарий (Бюджет = 2)
target_budget = 2
pred, certified, margin = model.certify_robustness(test_sample, budget=target_budget)

# Извеждане на текстов доклад в конзолата
print("--- ROBUSTNESS VERIFICATION REPORT ---")
print(f"Prediction: {'AI-Generated' if pred == 1 else 'Human-Authored'}")
print(f"Adversarial Budget: {target_budget} words")
print(f"Is Prediction Certified Robust? {'YES' if certified else 'NO'}")
print(f"Safe Margin (Stability): {margin:.4f}\n")

# 4. СИМУЛАЦИЯ: Тестване на устойчивостта при променлив бюджет на атака (0 до 10 думи)
budgets = np.arange(0, 11)
margins = []
certification_status = []

for b in budgets:
    _, is_certified, m = model.certify_robustness(test_sample, budget=b)
    margins.append(m)
    certification_status.append(is_certified)

# 5. ВИЗУАЛИЗАЦИЯ: Генериране и запазване на графика за GitHub профила Ви
plt.figure(figsize=(8, 5))
plt.plot(budgets, margins, label='Certified Margin', color='blue', linewidth=2, marker='o')
plt.axhline(0, color='red', linestyle='--', label='Robustness Threshold (Margin = 0)')

# Оцветяване на зоните (Зелено = Сертифициран/Безопасен, Червено = Уязвим)
fill_budget = budgets
plt.fill_between(fill_budget, margins, 0, where=(np.array(margins) > 0), color='green', alpha=0.15, label='Certified Safe Zone')
plt.fill_between(fill_budget, margins, 0, where=(np.array(margins) <= 0), color='red', alpha=0.15, label='Vulnerable Zone')

plt.title('Certified Robustness Verification Bounds', fontsize=12, fontweight='bold')
plt.xlabel('Adversarial Budget (Number of Manipulated Words)', fontsize=10)
plt.ylabel('Stability Margin (Log-Likelihood Difference)', fontsize=10)
plt.xticks(budgets)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

# Запазване на графиката като изображение
plt.savefig('robustness_plot.png', dpi=300, bbox_inches='tight')
print("Graph 'robustness_plot.png' has been successfully generated for your GitHub repository!")
