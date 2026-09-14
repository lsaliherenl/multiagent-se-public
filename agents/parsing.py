"""Paylaşılan LLM yanıt ayrıştırma yardımcıları (baseline/planner/coder ortak kullanır)."""

import json
import re


def extract_code(text: str) -> str:
    """Yanıttaki ```python``` bloklarından en uzununu alır; blok yoksa ham metni döndürür."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len).strip() if blocks else text.strip()


def extract_json(text: str):
    """Serbest metin içinden JSON değerini ayıklar (```json``` bloğu, ham metin
    ya da metne gömülü {...}). Ayrıştırılamazsa ValueError fırlatır — çağıran
    taraf (agents/validator.py) bunu bir sözleşme ihlali olarak ele alır.

    Not: son çare (gömülü {...}) basit find/rfind kullanır — metinde birden
    fazla ayrı JSON parçası varsa güvenilir değildir, ama planner'dan "sadece
    JSON döndür" istendiği için pratikte tek nesne bekleniyor.
    """
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    for candidate in (blocks or [text]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("yanıtta geçerli JSON bulunamadı")
