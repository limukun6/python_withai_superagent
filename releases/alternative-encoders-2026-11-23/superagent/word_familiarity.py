"""Estimate average token surprisal using only a fitted corpus of unique texts."""
from collections import Counter
from hashlib import sha256
import math
import re

import numpy as np


TOKENS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+|[^\w\s]", re.UNICODE)
UNKNOWN = '<UNK>'
START = '<BOS>'


class WordFamiliarity:
    def fit(self, texts):
        unique = set(texts)
        tokens = [TOKENS.findall(text.casefold()) for text in unique]
        counts = Counter(word for title in tokens for word in title)
        self.vocabulary_ = {word for word, count in counts.items() if count >= 2}
        self.title_digests_ = {sha256(text.encode()).digest() for text in unique}
        self.unigrams_, self.contexts_, self.bigrams_ = Counter(), Counter(), Counter()
        for title in tokens:
            previous = START
            for word in title:
                word = word if word in self.vocabulary_ else UNKNOWN
                self.unigrams_[word] += 1
                self.contexts_[previous] += 1
                self.bigrams_[previous, word] += 1
                previous = word
        self.total_ = sum(self.unigrams_.values())
        return self

    def score(self, text):
        words = [word if word in self.vocabulary_ else UNKNOWN
                 for word in TOKENS.findall(text.casefold())]
        if not words:
            return 0.0
        context = [START, *words[:-1]]
        pairs = list(zip(context, words))
        own = sha256(text.encode()).digest() in self.title_digests_
        own_words = Counter(words) if own else {}
        own_context = Counter(context) if own else {}
        own_pairs = Counter(pairs) if own else {}
        denominator = self.total_ - (len(words) if own else 0) + len(self.vocabulary_) + 1
        total = 0.0
        for previous, word in pairs:
            marginal = (self.unigrams_[word] - own_words.get(word, 0) + 1) / denominator
            matches = self.bigrams_[previous, word] - own_pairs.get((previous, word), 0)
            contexts = self.contexts_[previous] - own_context.get(previous, 0)
            total -= math.log((matches + 10 * marginal) / (contexts + 10))
        return total / len(words)

    def differences(self, rows):
        scores = {text: self.score(text) for text in {t for pair in rows for t in pair}}
        return np.asarray([scores[left] - scores[right] for left, right in rows], dtype=float)
