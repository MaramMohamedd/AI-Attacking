## AI Attacks and Defenses on SMS Spam Detection

### Project Overview

This project explores two types of adversarial attacks on an SMS spam detection system using Logistic Regression with TF-IDF vectorization:

1. **Mad-Lib Attack** - Synonym substitution attack using WordNet
2. **FGSM Attack (Fast Gradient Sign Method)** - Gradient-based attack on TF-IDF vectors

The project demonstrates both attack effectiveness and defense mechanisms, particularly adversarial training.

---

### Attacks Implemented

#### Mad-Lib Attack

**How it works:** The attack replaces words with their synonyms using WordNet, making the spam filter unable to recognize the message.

| Original Spam | After Mad-Lib Attack |
|---------------|----------------------|
| "Congratulations! You won a free iPhone. Click here to claim." | "Kudos! You earned a complimentary smartphone. Tap this link to receive." |

The spam filter sees completely different words, causing it to fail.

#### FGSM Attack (Fast Gradient Sign Method)

**Mathematical Formula:**
```
X_adv = X + ε · sign(∇_X L(θ, X, y))
```

**Where:**
- `X` = original input
- `X_adv` = adversarial example
- `ε` (epsilon) = perturbation magnitude (attack strength)
- `sign()` = function that returns -1, 0, or 1
- `∇_X L` = gradient of the loss function with respect to the input

**Step-by-step process:**
1. Calculate the gradient (how much each feature contributes to prediction)
2. Determine direction using sign() function
3. Apply perturbation by adding/subtracting ε from each feature
4. Result pushes the model toward a wrong prediction

> **Note:** Unlike Mad-Lib (works on raw text), FGSM works on TF-IDF vector representation and perturbs numerical values.

---

### Defense Strategies

#### Defense 1: Character-Level Features
- Use character n-grams (sequences of 2-4 characters) instead of words
- Why it works: "iPhone" → "smartphone" still contains overlapping sequences like "pho", "one"

#### Defense 2: Adversarial Training
Steps:
1. Generate attacked samples using FGSM with true labels
2. Augment original data with attacked samples
3. Retrain Logistic Regression + TF-IDF on augmented data
4. Test model accuracy on attacked samples

---

### Experimental Pipeline

| Phase | Description |
|-------|-------------|
| **Phase 1: Baseline** | Train Logistic Regression + TF-IDF on clean data, record accuracy on clean and attacked test sets |
| **Phase 2: Attack Demonstration** | Apply Mad-Lib attack to test messages, measure accuracy drop (94% → 50-60%) |
| **Phase 3: Defense Evaluation** | Implement character n-grams and/or adversarial training, measure accuracy recovery |

---

### Results

#### Final Performance Metrics

| Model | Clean Accuracy | Under Attack |
|-------|----------------|--------------|
| Original Model | 97.2% | 88.8% |
| Robust Model (After Defense) | 97.5% | 89.8% |

**Defense Effectiveness:**
- Undefended under attack: **88.8%**
- Defended under attack: **89.8%**
- Improvement: **+1.0%**

#### Key Findings from Parameter Tuning

| Parameter | Effect |
|-----------|--------|
| **Epsilon (ε=0.5)** | Model severely affected, harder to defend |
| **Epsilon (ε=0.1)** | Less attacking effect, better defense |
| **Augmentation Ratio (50%)** | Overfitting (-1.8% accuracy) |
| **Augmentation Ratio (15%)** | Balanced learning, positive improvement |
| **Multiple iterations** | Progressive robustness gain |

**Optimal Configuration Found:**
- Epsilon = 0.1
- Augmentation ratio = 15%
- Iterations = 10

---

### Why 97% Accuracy Is Not Overfitting

1. **TF-IDF as an equalizer:** Down-weights common words ("the", "a") and up-weights rare, informative spam words

2. **Precision & recall validation:** Model showed >90% precision and recall for spam class (not just high accuracy)

3. **Dummy classifier check:** A model always predicting "ham" would achieve only ~87% accuracy. Our 97% is 10% better than naive guess

---

### How Defenses Work

1. **Character n-grams** capture sub-word patterns that synonyms cannot hide
2. **Adversarial training** teaches the model to recognize attacked patterns
3. **Combining both** provides robust protection

---

### Conclusion

After multiple parameter trials, we found that:
- Epsilon value significantly affects defense success
- High augmentation ratio (50%) causes overfitting
- Optimal parameters (ε=0.1, ratio=15%, iterations=10) achieved +1.0% improvement under attack while maintaining clean accuracy

The adversarial training approach successfully made the model more robust against FGSM attacks without sacrificing performance on clean data.

---

### Files in This Project

| File | Description |
|------|-------------|
| `mid_lib_attack.ipynb` | Mad-Lib attack implementation |
| `fgsm_attack.ipynb` | FGSM attack and adversarial training |
| `README.md` | This file |

---

### Requirements

```
numpy
pandas
scikit-learn
nltk
scipy
matplotlib
```

---

### Author

Maram - AI Security Course Project

- SMS Spam Collection dataset
- WordNet for synonym generation
- Fast Gradient Sign Method (Goodfellow et al., 2014)
