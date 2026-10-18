from findex.tokenize import tokenize


def test_mixed_case():
    assert list(tokenize("Hello WORLD")) == ["hello", "world"]


def test_cyrillic():
    assert list(tokenize("Привіт, світе!")) == ["привіт", "світе"]


def test_apostrophes():
    assert list(tokenize("don't tell me about п'ять м’ячів")) == [
        "don't",
        "tell",
        "me",
        "about",
        "п'ять",
        "м’ячів",
    ]


def test_combining_mark():
    decomposed = "cafe\u0301"
    precomposed = "café"
    assert list(tokenize(decomposed)) == ["café"]
    assert list(tokenize(decomposed)) == list(tokenize(precomposed))


def test_punctuation_and_hyphens():
    assert list(tokenize("state-of-the-art... really?!")) == [
        "state",
        "of",
        "the",
        "art",
        "really",
    ]


def test_empty_string():
    assert list(tokenize("")) == []
    assert list(tokenize("   \n\t   ")) == []


def test_casefold_special():
    assert list(tokenize("Große Straße")) == ["grosse", "strasse"]


def test_digits():
    assert list(tokenize("python3 in 2026")) == ["python3", "in", "2026"]
