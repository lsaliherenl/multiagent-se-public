"""agents.parsing birim testleri: kod ve JSON ayıklama, LLM'siz."""

import pytest

from agents.parsing import extract_code, extract_json


def test_extract_code_python_etiketli():
    assert extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"


def test_extract_json_fenced_blok():
    text = '```json\n{"a": 1, "b": [1, 2]}\n```'
    assert extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_etiketsiz_fenced():
    text = '```\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_extract_json_ham_metin():
    assert extract_json('{"x": true}') == {"x": True}


def test_extract_json_etraf_metinli():
    text = 'Here is the plan:\n{"x": 1, "y": {"z": 2}}\nHope this helps!'
    assert extract_json(text) == {"x": 1, "y": {"z": 2}}


def test_extract_json_gecersiz_hata_firlatir():
    with pytest.raises(ValueError):
        extract_json("bu hic json degil")
