"""Harness birim testleri: hata sınıflandırma + görev seti bütünlüğü.

test_referans_cozum_gecer, 20 görevin tamamının referans çözümünü gerçek
harness yolundan geçirir — hem görev dönüşümünü hem değerlendirme zincirini
LLM'siz doğrular (harness'a güvenemezsek deney sonuçlarına da güvenemeyiz).
"""

import pytest

from eval.harness import evaluate, load_all_tasks

# Sentetik mini görev: harness'ın sınıflandırma mantığını izole test etmek için
DUMMY_TASK = {
    "task_id": "dummy",
    "entry_point": "topla",
    "test_code": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
}


def test_dogru_cozum_passed():
    r = evaluate(DUMMY_TASK, "def topla(a, b):\n    return a + b\n")
    assert r.status == "passed"
    assert r.error_class is None
    assert r.traceback is None


def test_yanlis_cozum_assertion():
    r = evaluate(DUMMY_TASK, "def topla(a, b):\n    return a - b\n")
    assert r.status == "failed"
    assert r.error_class == "assertion"
    assert "AssertionError" in r.traceback


def test_bozuk_sozdizimi_syntax():
    r = evaluate(DUMMY_TASK, "def topla(a, b)\n    return a + b\n")
    assert r.status == "failed"
    assert r.error_class == "syntax"


def test_eksik_fonksiyon_runtime():
    r = evaluate(DUMMY_TASK, "def baska_ad(a, b):\n    return a + b\n")
    assert r.status == "failed"
    assert r.error_class == "runtime"
    assert "NameError" in r.traceback


def test_sonsuz_dongu_timeout():
    r = evaluate(DUMMY_TASK, "def topla(a, b):\n    while True: pass\n", timeout_s=2)
    assert r.status == "timeout"
    assert r.error_class == "timeout"


def test_gorev_seti_yuklu_ve_20_gorev():
    tasks = load_all_tasks()
    assert len(tasks) == 20
    assert all(t["prompt"] and t["entry_point"] and t["test_code"] for t in tasks)


@pytest.mark.parametrize("task", load_all_tasks(), ids=lambda t: t["task_id"])
def test_referans_cozum_gecer(task):
    r = evaluate(task, task["reference_solution"])
    assert r.status == "passed", f"{task['task_id']}: {r.error_class}\n{r.traceback}"
