# coding=utf-8
"""Measure how much of the emotion label leaks through the input text alone.

If a plain bag-of-character-ngrams classifier can recover the emotion from the
text you feed the TTS model, then the emotion embedding table has no reason to
learn anything -- gradient descent will read the text instead. That is the
confound to rule out before touching the model architecture.

Usage:
    python3 finetuning/check_text_leakage.py <train.jsonl> <val.jsonl>

Needs only scikit-learn. Works on Japanese without a tokenizer because it uses
character n-grams rather than words.
"""
import json
import sys
from collections import Counter

# Japanese output dies on a cp932/cp950 Windows console otherwise.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("emotion"):
                rows.append((row["text"], row["emotion"]))
    if not rows:
        raise SystemExit(f"No rows with an 'emotion' field in {path}")
    return zip(*rows)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    train_text, train_y = load(sys.argv[1])
    val_text, val_y = load(sys.argv[2])

    labels = sorted(set(train_y))
    chance = 1.0 / len(labels)
    majority = Counter(val_y).most_common(1)[0][1] / len(val_y)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)

    classifier = LogisticRegression(max_iter=2000, C=5.0)
    classifier.fit(x_train, train_y)
    predicted = classifier.predict(x_val)
    accuracy = accuracy_score(val_y, predicted)

    print(f"train {x_train.shape[0]} / val {x_val.shape[0]} | {len(labels)} emotions")
    print(f"chance          {chance:.3f}")
    print(f"majority class  {majority:.3f}")
    print(f"TEXT-ONLY ACC   {accuracy:.3f}   <-- the leak")
    print()
    print(classification_report(val_y, predicted, zero_division=0))

    print("Most emotion-predictive character n-grams still in your text:")
    feature_names = np.array(vectorizer.get_feature_names_out())
    for index, label in enumerate(classifier.classes_):
        weights = classifier.coef_[index]
        top = feature_names[np.argsort(weights)[-12:][::-1]]
        joined = "  ".join(t.strip() for t in top if t.strip())
        print(f"  {label:<9} {joined}")

    print()
    if accuracy > 0.5:
        print("VERDICT: the text still carries the emotion. Fixing the model will not")
        print("         help -- the emotion vectors have nothing left to explain.")
    elif accuracy > chance * 2:
        print("VERDICT: partial leak. Worth reducing further, but no longer the whole story.")
    else:
        print("VERDICT: text is roughly emotion-neutral. The bottleneck is elsewhere")
        print("         (capacity / injection point) -- see the architecture options.")


if __name__ == "__main__":
    main()
