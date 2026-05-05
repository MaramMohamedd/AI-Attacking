import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

CHAR_SUBSTITUTIONS = {
    'a': '@', 'o': '0', 'i': '1', 'e': '3',
    's': '$', 't': '7', 'l': '1', 'g': '9',
    'b': '6', 'z': '2', 'h': '#'
}
DECOY_WORDS = ['hello', 'friend', 'family', 'good', 'morning',
               'thanks', 'please', 'kindly', 'happy', 'nice']

def perturb_text(text, rate=0.8, seed=42):
    """
    Strong evasion attack:
    1. Replace characters with symbols (rate=0.8 = 80% of words)
    2. Insert random innocent words to confuse the model
    """
    random.seed(seed)
    words = text.split()
    perturbed = []

    for word in words:
        if random.random() < rate:
            
            new_word = ""
            for ch in word:
                if ch in CHAR_SUBSTITUTIONS:
                    new_word += CHAR_SUBSTITUTIONS[ch]
                else:
                    new_word += ch
            perturbed.append(new_word)
        else:
            perturbed.append(word)

        if random.random() < 0.3:
            perturbed.append(random.choice(DECOY_WORDS))

    return " ".join(perturbed)

def evasion_attack(model, vectorizer, df_spam, n_examples=5):
    spam_df = df_spam[df_spam['label'] == 1].copy().reset_index(drop=True)

    original_texts  = spam_df['clean_text'].tolist()
    perturbed_texts = [perturb_text(t) for t in original_texts]

    X_original  = vectorizer.transform(original_texts)
    X_perturbed = vectorizer.transform(perturbed_texts)

    pred_original  = model.predict(X_original)
    pred_perturbed = model.predict(X_perturbed)

    evaded = sum(pred_perturbed == 0)
    total  = len(pred_perturbed)

    print(f"Evasion Attack Results:")
    print(f"  Total spam samples:         {total}")
    print(f"  Correctly caught (before):  {sum(pred_original == 1)}")
    print(f"  Evaded detection (after):   {evaded}")
    print(f"  Evasion rate:               {evaded/total*100:.1f}%")

    print(f"\n--- {n_examples} Example Perturbations ---")
    for i in range(min(n_examples, len(original_texts))):
        orig = original_texts[i]
        pert = perturbed_texts[i]
        label_orig = "SPAM caught" if pred_original[i] == 1 else "missed"
        label_pert = "SPAM caught" if pred_perturbed[i] == 1 else "EVADED"
        print(f"\n[{i+1}] Original  ({label_orig}): {orig[:80]}")
        print(f"    Perturbed ({label_pert}): {pert[:80]}")

    return pred_original, pred_perturbed, spam_df['label'].values