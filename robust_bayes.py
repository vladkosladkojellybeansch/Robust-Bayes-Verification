import numpy as np

class CertifiedRobustBayes:
    """
    Implementation of a Naive Bayes Classifier with Certified Robustness guarantees.
    Calculates the safety margin against adversarial word-substitution attacks.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.log_prior = None
        self.log_likelihood = None
        self.vocab_size = 0
        
    def fit(self, X, y):
        """
        Train the model.
        X: Document-term matrix (counts)
        y: Binary labels (0: Human, 1: AI-generated)
        """
        n_samples, self.vocab_size = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.log_prior = np.log(np.bincount(y) / n_samples)
        counts = np.zeros((n_classes, self.vocab_size))
        
        for c in self.classes:
            counts[c] = X[y == c].sum(axis=0)
            
        # Laplace Smoothing for numerical stability
        self.log_likelihood = np.log((counts + self.alpha) / 
                                      (counts.sum(axis=1, keepdims=True) + self.alpha * self.vocab_size))

    def certify_robustness(self, x, budget=1):
        """
        CORE LOGIC: Calculates if a prediction remains stable given an adversarial budget.
        An attacker can change up to 'budget' number of words.
        """
        # Linear decision function in log-space
        scores = self.log_prior + (x * self.log_likelihood).sum(axis=1)
        predicted_class = np.argmax(scores)
        other_class = 1 - predicted_class
        
        # Calculate the impact of each word on the classification margin
        # log(P(w|pos)) - log(P(w|neg))
        likelihood_diff = self.log_likelihood[predicted_class] - self.log_likelihood[other_class]
        
        # Worst-case scenario: attacker changes the most influential words
        # We take the absolute impact of the top 'budget' features
        sorted_impacts = np.sort(np.abs(likelihood_diff))[::-1]
        worst_case_loss = sorted_impacts[:budget].sum()
        
        margin = scores[predicted_class] - scores[other_class]
        is_certified = margin > worst_case_loss
        certified_margin = margin - worst_case_loss
        
        return predicted_class, is_certified, certified_margin
