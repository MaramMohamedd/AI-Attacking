from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np

REVERSE_SUBSTITUTIONS = {
    '@': 'a', '0': 'o', '1': 'i', '3': 'e',
    '$': 's', '7': 't', '9': 'g', '6': 'b',
    '2': 'z', '#': 'h'
}

DECOY_WORDS = {'hello', 'friend', 'family', 'good', 'morning',
               'thanks', 'please', 'kindly', 'happy', 'nice'}

def normalize_text(text):
    words = text.split()
    cleaned = []
    for word in words:
        if word in DECOY_WORDS:
            continue
        cleaned.append(''.join(REVERSE_SUBSTITUTIONS.get(ch, ch) for ch in word))
    return ' '.join(cleaned)

def defended_vectorize(texts, vectorizer):
    cleaned = [normalize_text(t) for t in texts]
    return vectorizer.transform(cleaned)

def ensemble_voting(df, vectorizer):
    X = vectorizer.transform(df['clean_text'])
    y = df['label']

    model1 = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
    model2 = MultinomialNB(alpha=0.1)
    model3 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)

    return (model1, model2, model3)