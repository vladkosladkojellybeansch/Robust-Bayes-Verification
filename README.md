# Robust-Bayes-Verification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized implementation of a Naive Bayes classifier that incorporates **certified robustness** principles. This project demonstrates how to provide mathematical guarantees for probabilistic classifiers against adversarial attacks.

## 🔬 Scientific Context
In the context of **Trustworthy Machine Learning**, empirical success is not enough. This project implements a certifier that calculates the "worst-case" scenario for word-substitution attacks. 

Inspired by the research on model verification (e.g., Lampert et al.), the classifier doesn't just predict a label; it verifies if the label remains stable within a defined **adversarial budget** ($K$).

## 🛠️ Features
- **Certified Bounds:** Calculates the margin between classes vs. the maximum possible perturbation.
- **Log-likelihood Analysis:** Uses high-precision log-space calculations for numerical stability.
- **Minimalist Implementation:** Built with pure NumPy to show the underlying algorithmic logic.

## 🚀 Quick Start
Run the benchmark to see the robustness certification in action:
```bash
python benchmark_run.py

````````````

## 📊 Methodology
The certifier validates the mathematical stability of the prediction by checking the following condition:

$$\Delta(x) > \sum_{i \in \mathcal{K}} \text{impact}(w_i)$$

**Where:**
- $\Delta(x)$ is the classification margin (the difference between the log-probability of the predicted class and the next best class).
- $\mathcal{K}$ represents the set of the top-$K$ most influential words (features) within the adversarial budget.
- $\text{impact}(w_i)$ is the absolute difference in log-likelihoods for word $i$ across classes.

If this condition holds, the model is **mathematically guaranteed** to maintain its prediction even if an attacker modifies any $K$ words in the input.
