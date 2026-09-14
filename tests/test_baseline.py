"""Baseline'ın LLM'siz parçaları: kod bloğu ayıklama."""

from pipeline.baseline import extract_code


def test_python_etiketli_blok():
    text = "Here is the solution:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(text) == "def f():\n    return 1"


def test_etiketsiz_blok():
    text = "```\ndef f():\n    return 2\n```"
    assert extract_code(text) == "def f():\n    return 2"


def test_birden_fazla_blokta_en_uzun():
    text = "```python\nx = 1\n```\naciklama\n```python\ndef f():\n    return 3\n```"
    assert extract_code(text) == "def f():\n    return 3"


def test_blok_yoksa_ham_metin():
    assert extract_code("def f():\n    return 4\n") == "def f():\n    return 4"
