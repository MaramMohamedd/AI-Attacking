import re
from sklearn.linear_model import LogisticRegression

REVERSE_SUBSTITUTIONS = {
    '@': 'a', '0': 'o', '1': 'i', '3': 'e',
    '$': 's', '7': 't'
}

def normalize_text(text):
    """Defense 1 — Input Sanitization"""
    result = ""
    for ch in text:
        result += REVERSE_SUBSTITUTIONS.get(ch, ch)
    return result

def defended_vectorize(texts, vectorizer):
    """Apply normalization before transforming."""
    cleaned = [normalize_text(t) for t in texts]
    return vectorizer.transform(cleaned)

def adversarial_training(df, vectorizer, perturb_fn, rate=0.3):
    """Defense 2 — Adversarial Training"""
    from attack_evasion import perturb_text
    import pandas as pd

    spam_df = df[df['label'] == 1].copy()

    # Two perturbation variants per spam message
    perturbed1 = spam_df.copy()
    perturbed1['clean_text'] = spam_df['clean_text'].apply(
        lambda t: perturb_text(t, rate=0.3, seed=1)
    )
    perturbed2 = spam_df.copy()
    perturbed2['clean_text'] = spam_df['clean_text'].apply(
        lambda t: perturb_text(t, rate=0.5, seed=2)
    )

    augmented_df = pd.concat([df, perturbed1, perturbed2], ignore_index=True)
    X_aug = vectorizer.transform(augmented_df['clean_text'])
    y_aug = augmented_df['label']

    defended_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )
    defended_model.fit(X_aug, y_aug)

    return defended_model