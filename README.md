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
