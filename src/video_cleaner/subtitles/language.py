"""Telling whether a subtitle's text is English, from its words alone.

A file's name can say nothing about its language (a ".de" or ".zh-Hans" tag is not always
there) and a provider can return the wrong file, so the resolver checks the content. A subtitle
in the wrong language is worse than none, so this fails safe: text that is too short to judge is
not English.

The method is a count, with no dependency and no model. The text is lower-cased and split into
words (runs of Unicode letters, with apostrophes removed first so "don't" is one word, "dont",
and "it's" is "its"). The share of words that are among a few dozen of the most common English
words is compared with `COMMON_WORD_SHARE_MIN`. English dialogue has a large share of such
words, while other languages have almost none: the words that English shares with another
language ("a", "in", "on", "no", "me", "so", "do", "is", "was", "die") are left out of the set,
because "was", "so" and "in" are very common in German and "a", "no" and "me" in Spanish and
Italian. Languages written without spaces (Chinese, Japanese) give a few long "words" and no
common ones. Only the first `MAX_CUES` cues and `MAX_CHARS` characters are read, so the cost
does not depend on the size of the file.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from video_cleaner.subtitles.textfmt import Cue

#: At least this many words are needed to judge; fewer is "cannot tell", which is not English.
MIN_WORDS = 50
#: The share of words that must be common English words. The English samples in
#: tests/subtitles/test_language.py measure 0.27 to 0.62 and the German, French, Spanish, Italian,
#: Russian, Chinese and Japanese ones 0.00 to 0.03, so this sits between with room on both sides.
COMMON_WORD_SHARE_MIN = 0.15
#: Most cues, and most characters of their text, that are read.
MAX_CUES = 2_000
MAX_CHARS = 200_000

# A word is a run of letters: not a digit, an underscore or anything that is not a word
# character. The pattern is one plain greedy run, so it matches in linear time.
_WORD_RE = re.compile(r"[^\W\d_]+")
# The ASCII apostrophe, the two curly single quotes, the modifier letter apostrophe, a backtick.
_APOSTROPHES = str.maketrans("", "", "".join(map(chr, (0x27, 0x2019, 0x2018, 0x2BC, 0x60))))
# Very common English words (apostrophes removed from the contractions), none of them a common
# word of German, French, Spanish, Italian or Russian. A few ("to", "for", "of", "my") are common
# words of Polish, Danish or Dutch, but only one or two among many, so those languages stay far
# below the threshold.
_COMMON_WORDS = frozenset(
    {
        "the",
        "and",
        "you",
        "to",
        "of",
        "that",
        "it",
        "for",
        "with",
        "he",
        "she",
        "this",
        "not",
        "are",
        "be",
        "have",
        "but",
        "what",
        "we",
        "my",
        "your",
        "they",
        "there",
        "know",
        "can",
        "just",
        "would",
        "his",
        "her",
        "him",
        "from",
        "all",
        "about",
        "get",
        "how",
        "why",
        "who",
        "were",
        "been",
        "its",
        "dont",
        "thats",
        "youre",
        "ill",
        "ive",
        "cant",
        "didnt",
        "isnt",
        "had",
        "when",
        "then",
        "them",
        "out",
        "up",
        "if",
        "or",
        "one",
        "more",
        "now",
    }
)


def common_word_share(cues: Sequence[Cue]) -> tuple[int, float]:
    """(words read, share of them that are common English words) for the start of *cues*.

    Reads at most `MAX_CUES` cues and `MAX_CHARS` characters. The share is 0.0 when there are
    no words.
    """
    parts: list[str] = []
    budget = MAX_CHARS
    for cue in cues[:MAX_CUES]:
        if budget <= 0:
            break
        parts.append(cue.text[:budget])
        budget -= len(cue.text) + 1  # the newline that separates the cues
    text = "\n".join(parts).lower().translate(_APOSTROPHES)
    words = _WORD_RE.findall(text)
    if not words:
        return 0, 0.0
    common = sum(1 for word in words if word in _COMMON_WORDS)
    return len(words), common / len(words)


def looks_english(cues: Sequence[Cue]) -> bool:
    """Whether the text of *cues* is English: at least `MIN_WORDS` words, and at least
    `COMMON_WORD_SHARE_MIN` of them common English words. Pure and bounded; never raises.
    """
    words, share = common_word_share(cues)
    return words >= MIN_WORDS and share >= COMMON_WORD_SHARE_MIN
