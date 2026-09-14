"""Görev setini üretir: HumanEval + sanitized-MBPP ham verisini indirir,
sabit ID listesindeki görevleri ortak şemaya dönüştürüp tasks/*.json yazar.

Kullanım:  uv run python scripts/fetch_tasks.py

Determinizm: seçim listeleri bu dosyada sabittir; script her çalıştığında
aynı görev seti birebir yeniden üretilir (tekrarlanabilirlik şartı).
"""

import gzip
import hashlib
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TASKS_DIR

# master yerine commit'e sabitlendi (2026-07-21) — üç bağımsız doğrulama:
# GitHub commits API + iki ayrı canlı indirip hash hesaplama. Commit-pinning
# zaten içerik-adresleme garantisi veriyor; checksum'ın marjinal değeri
# bozuk/kesik indirmeyi ve COMMIT'in SHA256 güncellenmeden değişmesini
# yakalamak (abartılmayacak bir güvence — ikisi de nadir senaryolar).
HUMANEVAL_COMMIT = "463c980b59e818ace59f6f9803cd92c749ceae61"
MBPP_COMMIT = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
HUMANEVAL_URL = f"https://raw.githubusercontent.com/openai/human-eval/{HUMANEVAL_COMMIT}/data/HumanEval.jsonl.gz"
MBPP_URL = f"https://raw.githubusercontent.com/google-research/google-research/{MBPP_COMMIT}/mbpp/sanitized-mbpp.json"
HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"  # 44877 bayt
MBPP_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"       # 255053 bayt

# Seçim: kolay/orta/zor karışımı. HumanEval'de 39/109/115/129 zor uçta;
# MBPP sanitized (Austin et al. elle doğrulanmış alt küme) kolay-orta ağırlıklı.
# MBPP 56 bilinçli DIŞARIDA: fonksiyon adı `check`, harness'ın check(candidate)
# sözleşmesiyle çakışıyor.
HUMANEVAL_IDS = [0, 7, 12, 26, 31, 33, 39, 64, 96, 109, 115, 129]
MBPP_IDS = [2, 3, 4, 11, 14, 16, 19, 77]


def _verify_checksum(data: bytes, expected_sha256: str, url: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"checksum uyuşmazlığı ({url}): beklenen {expected_sha256}, alınan {actual}")


def _download(url: str, expected_sha256: str | None = None) -> bytes:
    print(f"indiriliyor: {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    if expected_sha256:
        _verify_checksum(data, expected_sha256, url)
    return data


def _extract_entry_point(code: str, test_example: str) -> str:
    """Referans çözümdeki def'lerden, örnek testte çağrılanı bulur.

    Assert satırından ad çıkarmak güvenilir değil (örn. `assert set(f(...))`),
    bu yüzden ters yönden gidilir: kodda tanımlı adlar test metninde aranır.
    """
    defined = re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code, flags=re.M)
    called = [name for name in defined if re.search(rf"\b{re.escape(name)}\s*\(", test_example)]
    if len(called) != 1:
        raise ValueError(f"entry point tekil bulunamadı: defined={defined}, called={called}")
    return called[0]


def convert_humaneval(raw: bytes) -> list[dict]:
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
        problems = {json.loads(line)["task_id"]: json.loads(line) for line in f if line.strip()}
    tasks = []
    for num in HUMANEVAL_IDS:
        p = problems[f"HumanEval/{num}"]
        tasks.append({
            "task_id": f"humaneval_{num:03d}",
            "source": "humaneval",
            "prompt": p["prompt"],
            "entry_point": p["entry_point"],
            # HumanEval'in test alanı zaten `def check(candidate)` tanımlar;
            # harness sonuna check(<entry_point>) çağrısını ekler.
            "test_code": p["test"],
            "reference_solution": p["prompt"] + p["canonical_solution"],
        })
    return tasks


def convert_mbpp(raw: bytes) -> list[dict]:
    problems = {t["task_id"]: t for t in json.loads(raw.decode("utf-8"))}
    tasks = []
    for num in MBPP_IDS:
        p = problems[num]
        entry_point = _extract_entry_point(p["code"], p["test_list"][0])
        # Assert'lerdeki fonksiyon adı `candidate` ile değiştirilerek HumanEval
        # ile aynı check(candidate) sözleşmesine getirilir. `is_Diff (x)` gibi
        # ad ile parantez arasında boşluk olan çağrılar da yakalanır.
        call_pat = re.compile(rf"\b{re.escape(entry_point)}\s*\(")
        body = "\n".join(f"    {call_pat.sub('candidate(', line)}" for line in p["test_list"])
        imports = "\n".join(p["test_imports"])
        test_code = (imports + "\n\n" if imports else "") + "def check(candidate):\n" + body + "\n"
        prompt = (
            f"{p['prompt'].strip()}\n\n"
            f"Write a Python function named `{entry_point}`.\n"
            f"Example test:\n{p['test_list'][0]}\n"
        )
        tasks.append({
            "task_id": f"mbpp_{num:03d}",
            "source": "mbpp",
            "prompt": prompt,
            "entry_point": entry_point,
            "test_code": test_code,
            "reference_solution": p["code"],
        })
    return tasks


def main() -> None:
    tasks = (convert_humaneval(_download(HUMANEVAL_URL, HUMANEVAL_SHA256))
             + convert_mbpp(_download(MBPP_URL, MBPP_SHA256)))
    TASKS_DIR.mkdir(exist_ok=True)
    for task in tasks:
        path = TASKS_DIR / f"{task['task_id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(tasks)} görev yazıldı -> {TASKS_DIR}")
    print("  humaneval:", sum(t["source"] == "humaneval" for t in tasks),
          "| mbpp:", sum(t["source"] == "mbpp" for t in tasks))


if __name__ == "__main__":
    main()
